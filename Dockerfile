FROM nvcr.io/nvidia/pytorch:24.03-py3

# System-Dependencies (X-Libs + Prolog)
RUN apt update && \
    apt install -y \
      libxxf86vm1 libx11-6 libxi6 libxfixes3 libxrandr2 \
      libxrender1 libxt6 xvfb swi-prolog git wget && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Blender
RUN wget https://download.blender.org/release/Blender4.4/blender-4.4.0-linux-x64.tar.xz && \
    tar -xf blender-4.4.0-linux-x64.tar.xz -C /usr/local/ && \
    ln -s /usr/local/blender-4.4.0-linux-x64/blender /usr/local/bin/blender && \
    rm blender-4.4.0-linux-x64.tar.xz

# Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

# Conda-Env
WORKDIR /workspace
COPY environment.yml .
RUN conda env create -n zendo -f environment-complete.yml && conda clean --all -y
SHELL ["conda", "run", "-n", "zendo", "/bin/bash", "-c"]

# Optional: Repo gleich mit rein
# COPY . /workspace/zendo-model
# WORKDIR /workspace/zendo-model

CMD ["bash"]
