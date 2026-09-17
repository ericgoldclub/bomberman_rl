FROM continuumio/miniconda3
WORKDIR /home/bomberman
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*
RUN conda install -y --override-channels -c conda-forge \
    python=3.11 scipy numpy matplotlib numba
RUN conda install -y --override-channels -c pytorch -c conda-forge \
    pytorch torchvision
RUN pip install scikit-learn tqdm tensorflow keras tensorboardX xgboost lightgbm
RUN pip install pathfinding pyaml igraph ujson
RUN conda install -y --override-channels -c conda-forge pandas
RUN pip install networkx dill pyastar2d easydict sympy pygame
COPY . .
CMD ["/bin/bash"]
