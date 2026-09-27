#!/usr/bin/env python3
"""Assemble the final report from the project sections."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
WRITEUP = ROOT / 'Writeup'


def extract(text, start, end, after=0):
    first = text.index(start, after)
    last = text.index(end, first + len(start))
    return text[first:last].strip()


def attribute(text, author):
    return re.sub(r'(?m)^(\\(?:section|subsection|subsubsection)\{[^\n]*\})([^\n]*)$',
                  lambda match: match.group(0) + '\n' + r'\contributionauthor{' + author + '}', text)


def build():
    original = WRITEUP / 'main.tex'
    original_hash = hashlib.sha256(original.read_bytes()).hexdigest()
    provenance_path = WRITEUP / 'experiments/results/report_author_provenance.json'
    provenance = json.loads(provenance_path.read_text())
    if provenance['source_main_sha256'] != original_hash:
        raise RuntimeError('Original report changed; recheck author provenance before building')
    with tempfile.TemporaryDirectory() as name:
        temporary = Path(name)
        (temporary / 'Writeup').mkdir()
        (temporary / 'Writeup/main.tex').write_bytes(original.read_bytes())
        subprocess.run(['patch', '--silent', '-p1', '-d', str(temporary),
                        '-i', str(WRITEUP / 'scientific_corrections.patch')], check=True)
        source = (temporary / 'Writeup/main.tex').read_text()
    background = extract(source, r'\subsection{Bomberman', r'\subsection{Input representation')
    features = extract(source, r'\subsection{Input representation', r'\subsection{Exploration')
    features = features.replace(r'\subsection{Input representation', r'\subsubsection{Input representation', 1)
    exploration = extract(source, r'\subsection{Exploration', r'\subsection{Ruehl based agent}')
    architecture = extract(source, r'\subsubsection{Neural network architecture}', r'\subsubsection{Training curriculum}')
    curriculum = extract(source, r'\subsubsection{Training curriculum}', r'\subsubsection{Action masking')
    mask_rewards = extract(source, r'\subsubsection{Action masking', r'\subsubsection{Optimization')
    optimization = extract(source, r'\subsubsection{Optimization', r'\subsection{The Final Agent}')
    final = extract(source, r'\subsection{The Final Agent}', r'\section{Results and Discussion}')
    final = final.replace(r'\subsection{The Final Agent}', r'\subsection{Final agent}\label{sec:final-methods}', 1)
    historical = extract(source, r'\subsection{Ruehl based agent}', r'\section{Conclusion}',
                         source.index(r'\section{Results and Discussion}'))
    historical = historical.replace(r'\subsection{Ruehl based agent}', r'\subsection{Historical RUEHL selection summaries}', 1)
    historical = historical.replace(r'\subsection{Final agent}', r'\subsection{Illustrative final-agent training run}', 1)
    # Give each tall archived learning figure a full page for legibility.
    old_figure = r'''
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.49\textwidth]{figures/final_agent_loot_crate_learning_curves.png}%
    \hfill
    \includegraphics[width=0.49\textwidth]{figures/final_agent_learning_curves_classic.png}
    \caption{Learning curves of the final agent: loot-crate without opponents
    (left) and classic with three rule-based opponents (right).}
    \label{fig:final-agent-learning-curves}
\end{figure}
'''
    replacement_figure = r'''
\begin{figure}[p]
\centering
\includegraphics[height=0.88\textheight,width=\textwidth,keepaspectratio]{figures/final_agent_loot_crate_learning_curves.png}
\caption{Archived final-agent training statistics in loot-crate without opponents. Curves are ten-round rolling averages during learning, rather than independent frozen-policy evaluations.}
\label{fig:final-agent-learning-curves-loot}
\end{figure}
\begin{figure}[p]
\centering
\includegraphics[height=0.88\textheight,width=\textwidth,keepaspectratio]{figures/final_agent_learning_curves_classic.png}
\caption{Archived final-agent training statistics in classic with three rule-based opponents. The final panel averages steps only over survived rounds; it is not average episode duration.}
\label{fig:final-agent-learning-curves}
\end{figure}
'''
    if old_figure.strip() not in historical:
        raise RuntimeError('Archived figure block no longer matches')
    historical = historical.replace(old_figure.strip(), replacement_figure.strip())
    historical = historical.replace('Figure \\ref{fig:final-agent-learning-curves} shows these plots.',
        'Figures~\\ref{fig:final-agent-learning-curves-loot} and~\\ref{fig:final-agent-learning-curves} show these plots.')
    conclusion = extract(source, r'\section{Conclusion}', r'\bibliographystyle{plainnat}')
    conclusion = conclusion.replace(r'\section{Conclusion}', r'\subsection{Development conclusions}', 1)
    conclusion = conclusion.replace('campaign below measures', 'campaign measures')
    preamble = source[:source.index(r'\begin{document}')]
    preamble = preamble.replace(r'\usepackage{placeins}', r'\usepackage{placeins,needspace,etoolbox}')
    preamble += r'''
\newcommand{\contributionauthor}[1]{{\small\textit{Main author: #1}\par}}
\newcommand{\contributionauthors}[1]{{\small\textit{Main authors: #1}\par}}
\newcommand{\contributioninput}[1]{\begingroup
\let\section\subsection\let\subsection\subsubsection\input{#1}\endgroup}
\widowpenalty=10000
\clubpenalty=10000
\pretocmd{\section}{\Needspace{10\baselineskip}}{}{}
\pretocmd{\subsection}{\Needspace{8\baselineskip}}{}{}
\pretocmd{\subsubsection}{\Needspace{8\baselineskip}}{}{}
'''
    opening = r'''
\begin{document}
\maketitle
\input{sections/environment}
\begin{abstract}
We develop two Double DQN Bomberman agents with different allocations of
learned decisions and explicit guidance. A 3,000-state engine-based audit
and 1,600 frozen-policy games distinguish escape feasibility from competitive
performance. With final-agent weights fixed, replacing the RUEHL mask
with the temporal mask adds 1.375 points per game against rule-based opponents; learned
ranking adds 0.855 points beyond temporal masked random play while increasing
self-destruction. A replayed opponent collision exposes why conditional
escape search cannot guarantee safety. Together, these experiments connect
engine-conditional escape feasibility with learned choices and competitive outcomes.
\end{abstract}
\input{sections/context}
\section{Background}\label{sec:theory}
\contributionauthor{Mark Salzmann}
'''
    planning_methods = r'''
\FloatBarrier
\input{sections/planning}
\section{Methods}\label{sec:methods}
\contributionauthors{Mark Salzmann, Eric Flämig, Jesper Eggers}
\subsection{RUEHL agent}\label{sec:ruehl-methods}
\contributionauthor{Mark Salzmann}
'''
    training = r'''
\FloatBarrier
\contributioninput{sections/evaluation}
\section{Training}\label{sec:training}
\contributionauthors{Mark Salzmann, Jesper Eggers}
'''
    ruehl_training = r'''
\subsection{RUEHL curriculum and optimization}
\contributionauthor{Mark Salzmann}
'''
    results = r'''
\FloatBarrier
\contributioninput{sections/final_training}
\section{Experiments and Results}\label{sec:results}
\contributionauthors{Mark Salzmann, Eric Flämig, Jesper Eggers}
'''
    measured_results = r'''
\FloatBarrier
\contributioninput{sections/mask_audit}
\contributioninput{sections/results}
\contributioninput{sections/components}
\contributioninput{sections/failures}
\contributioninput{sections/compute}
\section{Conclusion}
\contributionauthors{Mark Salzmann, Eric Flämig, Jesper Eggers}
'''
    ending = r'''
\contributioninput{sections/synthesis}
\FloatBarrier
\clearpage
\bibliographystyle{plainnat}
\bibliography{bib/references,bib/evaluation_references}
\end{document}
'''
    final_history_start = historical.index(r'\subsection{Illustrative final-agent training run}')
    historical = '\n\n'.join([
        attribute(historical[:final_history_start].rstrip(), 'Mark Salzmann'),
        attribute(historical[final_history_start:], 'Eric Flämig'),
    ])
    assembled = '\n\n'.join([preamble, opening, attribute(background, 'Mark Salzmann'), planning_methods,
        attribute(features, 'Mark Salzmann'), attribute(architecture, 'Mark Salzmann'),
        attribute(mask_rewards, 'Mark Salzmann'), attribute(final, 'Eric Flämig'),
        training, attribute(exploration, 'Mark Salzmann'), ruehl_training,
        attribute(curriculum, 'Mark Salzmann'), attribute(optimization, 'Mark Salzmann'),
        results, historical, measured_results,
        attribute(conclusion, 'Mark Salzmann; Final-agent discussion: Eric Flämig'), ending])
    assembled = re.sub(r'\\contributioninput\{([^}]+)\}',
        lambda match: r'\begingroup\let\section\subsection\let\subsection\subsubsection' + '\n' +
                      r'\input{' + match.group(1) + r'}\endgroup', assembled)
    # Replace manual paragraph line breaks outside tables/math with paragraphs.
    protected = 0
    lines = []
    for line in assembled.splitlines():
        starts = len(re.findall(r'\\begin\{(?:tabularx|tabular|align|aligned|cases|array)\}', line))
        ends = len(re.findall(r'\\end\{(?:tabularx|tabular|align|aligned|cases|array)\}', line))
        protected += starts
        if not protected:
            line = re.sub(r'\\{2,}[ \t]*$', '\n', line)
        lines.append(line)
        protected -= ends
    assembled = '\n'.join(lines) + '\n'
    output = WRITEUP / 'final_report.tex'
    output.write_text(assembled)
    manifest = {
        'purpose': 'Report assembly; original main.tex is unchanged; authors are assigned from verified Git history.',
        'source_main_sha256': original_hash,
        'patch_sha256': hashlib.sha256((WRITEUP/'scientific_corrections.patch').read_bytes()).hexdigest(),
        'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'report_source_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'required_chapters': ['Introduction', 'Background', 'Project planning', 'Methods', 'Training', 'Experiments and Results', 'Conclusion'],
        'author_provenance_sha256': hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
        'unknown_original_author_assignments': False,
        'author_assignment_basis': 'Git commit diffs and line-level blame; mixed chapters credit their contributing writers.',
        'original_source_unchanged': original_hash == hashlib.sha256(original.read_bytes()).hexdigest(),
    }
    (WRITEUP/'experiments/results/integration_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Built {output}; original source unchanged.')

if __name__ == '__main__':
    build()
