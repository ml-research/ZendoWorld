# Testing ZendoWorld in Docker

`Dockerfile` is based on the file in `../zendo-model-refactored`, retaining its NVIDIA PyTorch base image. It now copies `environments/complete.yml` and installs the Python dependencies needed by Blender's bundled interpreter. The image contains a Conda environment named `zendo`; the source checkout is mounted when creating the container.

## Start Docker

On macOS, launch Docker Desktop and wait for the engine to be ready:

```bash
open -a Docker
docker info
```

On a Linux machine with Docker Engine installed:

```bash
sudo systemctl start docker
docker info
```

References: [Docker Desktop on Mac](https://docs.docker.com/desktop/setup/install/mac-install/), [starting Docker Engine](https://docs.docker.com/engine/daemon/start/).

## Select an image

List images on the machine where you will run the container:

```bash
docker image ls
```

For your existing image, replace the example with its repository and tag:

```bash
IMAGE_NAME=your-existing-image:tag
```

To build a new image, run this from the ZendoWorld repository root on an x86_64 Linux machine:

```bash
docker build -f docker/Dockerfile -t zendoworld:dev .
IMAGE_NAME=zendoworld:dev
```

The build installs the environment but does not copy the source code or datasets. An existing image can also be reused with the mounted checkout, provided its environment has the required dependencies.

## Create a container

For GPU testing on a Linux host with the NVIDIA Container Toolkit configured, run from the repository root:

```bash
docker run -it --name zendoworld-test \
  --gpus all --shm-size=8g \
  --mount "type=bind,source=$(pwd),target=/workspace/ZendoWorld" \
  --workdir /workspace/ZendoWorld \
  "$IMAGE_NAME" bash
```

For CPU testing on an ARM Mac, use the x86_64 image under emulation:

```bash
docker run -it --name zendoworld-test \
  --platform linux/amd64 --shm-size=8g \
  --mount "type=bind,source=$(pwd),target=/workspace/ZendoWorld" \
  --workdir /workspace/ZendoWorld \
  "$IMAGE_NAME" bash
```

This image targets Linux x86_64. Docker Desktop on macOS does not provide NVIDIA CUDA passthrough through `--gpus`; use the Linux GPU host for GPU experiments. CPU emulation on the Mac can be slow.

The mount makes current source edits visible immediately and keeps outputs in the checkout. Docker images are stored per Docker engine, so an image on a remote server will not automatically appear in Docker Desktop on the Mac.

References: [container creation and GPU options](https://docs.docker.com/reference/cli/docker/container/run), [bind mounts](https://docs.docker.com/engine/storage/bind-mounts/), [Docker Desktop GPU passthrough](https://docs.docker.com/desktop/features/gpu/).

## Activate the environment and test

Inside the container:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate zendo
python -m pip install -e . --no-deps
python -m pip install pytest
export PYTHONHASHSEED=0

zendoworld run --help
python -m pytest -q
```

To check GPU access on the Linux host:

```bash
nvidia-smi
python -c 'import torch; print(torch.cuda.is_available())'
```

For an initial API-backed game, configure your OpenAI API key through the environment or the existing credential file, then run:

```bash
zendoworld run \
  --player fullgpt \
  --prompter-model gpt-5-mini \
  --dataset data/combined_dataset_short.pkl \
  --threads 1 --seeds 1 --tasks 0
```

For an older image, install any missing experiment dependencies into `zendo`. Updating the bind-mounted source does not change the image's installed dependencies.

## Reopen the same container

After leaving the shell with `exit`, restart it with:

```bash
docker start -ai zendoworld-test
```

Activate `zendo` again in the new shell. The container retains installed packages between starts. While it is already running, open another shell using `docker exec -it zendoworld-test bash`.

Reference: [restart a stopped container](https://docs.docker.com/reference/cli/docker/container/start/).
