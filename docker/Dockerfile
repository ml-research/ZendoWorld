FROM nvcr.io/nvidia/pytorch:24.03-py3

# System dependencies for headless Blender and Prolog
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && \
    apt install -y \
      libxxf86vm1 libx11-6 libxi6 libxfixes3 libxrandr2 \
      libxrender1 libxt6 libxkbcommon0 libsm6 libgl1 libegl1 \
      xvfb swi-prolog git wget bzip2 ca-certificates && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Blender
RUN wget https://download.blender.org/release/Blender4.4/blender-4.4.0-linux-x64.tar.xz && \
    tar -xf blender-4.4.0-linux-x64.tar.xz -C /usr/local/ && \
    ln -s /usr/local/blender-4.4.0-linux-x64/blender /usr/local/bin/blender && \
    rm blender-4.4.0-linux-x64.tar.xz && \
    /usr/local/blender-4.4.0-linux-x64/4.4/python/bin/python3.11 -m ensurepip --upgrade && \
    /usr/local/blender-4.4.0-linux-x64/4.4/python/bin/python3.11 -m pip install --no-cache-dir \
      pyyaml "numpy<2" pillow && \
    /usr/local/blender-4.4.0-linux-x64/4.4/python/bin/python3.11 -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu "torch==2.5.1"

# Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

# Conda-Env
WORKDIR /workspace
COPY environments/complete.yml ./environments/complete.yml
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda env create -n zendo -f environments/complete.yml && \
    conda clean --all -y
SHELL ["conda", "run", "-n", "zendo", "/bin/bash", "-c"]

# Mount this checkout at /workspace/ZendoWorld when starting the container.
# Activate the zendo environment and install the package with pip install -e . --no-deps.
# SHELL controls build steps only; an interactive bash still needs conda activate.

CMD ["bash"]
