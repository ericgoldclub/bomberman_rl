from pathlib import Path
import matplotlib.pyplot as plt

log_path = Path("agent_code/final_agent/logs/final_agent_loot_crate.log")

# Separate Logs: "all". Gemischte Logs: "loot_crate" oder "classic".
PLOT_PHASE = "all"
LOOT_ROUNDS = 300  # Runden vor dem Wechsel von Loot-Crate zu Classic.

rows = []

with log_path.open() as log:
    for line in log:
        if "[ROLLING_STATS] " not in line:
            continue

        payload = line.split("[ROLLING_STATS] ", 1)[1]

        # Repair the missing separator in older log entries.
        payload = payload.replace("suicide_ratio=", " suicide_ratio=")

        fields = dict(item.split("=", 1) for item in payload.split())
        rows.append({key: float(value) for key, value in fields.items()})
      

if not rows:
    raise ValueError("No ROLLING_STATS entries found.")

# Find the most recent restart of the step counter.
start_index = 0

for i in range(1, len(rows)):
    if rows[i]["steps"] <= rows[i - 1]["steps"]:
        start_index = i

rows = rows[start_index:]

# Bei gemischten Logs nur Fenster behalten, die ganz in der gewählten Phase liegen.
if PLOT_PHASE == "loot_crate":
    rows = [row for row in rows if int(row["round"]) <= LOOT_ROUNDS]
elif PLOT_PHASE == "classic":
    rows = [
        row for row in rows
        if int(row["round"]) - int(row["window"]) + 1 > LOOT_ROUNDS
    ]
elif PLOT_PHASE != "all":
    raise ValueError("PLOT_PHASE muss 'all', 'loot_crate' oder 'classic' sein.")

if not rows:
    raise ValueError(
        f"Keine Mittelwerte für Phase '{PLOT_PHASE}' vorhanden. "
        "Für separate Logs PLOT_PHASE = 'all' verwenden; "
        "bei gemischten Logs LOOT_ROUNDS prüfen."
    )

for row in rows:
    row["avg_score"] = row["avg_coins"] + 5 * row["avg_kills"]


metrics = [
    ("avg_coins", "Average coins"),
    ("avg_kills", "Average kills"),
    ("avg_score", "Average score"),
    ("avg_crates", "Average crates"),
    ("suicide_ratio", "Suicide ratio"),
    ("avg_survived_steps", "Average steps in survived rounds"),
]

steps = [row["steps"] for row in rows]
fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 2.4 * len(metrics)), sharex=True)

for ax, (key, label) in zip(axes, metrics):
    ax.plot(steps, [row[key] for row in rows])
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)
    if key == "suicide_ratio":
        ax.set_ylim(0, 1)

axes[-1].set_xlabel("Training steps since process start")
phase_title = "" if PLOT_PHASE == "all" else f" in {PLOT_PHASE.replace('_', '-')}"
fig.suptitle(f"Rolling averages{phase_title}")
fig.tight_layout()
output_name = (
    f"{log_path.stem}_learning_curves.png" if PLOT_PHASE == "all"
    else f"final_agent_learning_curves_{PLOT_PHASE}.png"
)
fig.savefig(output_name, dpi=200)
plt.show()
