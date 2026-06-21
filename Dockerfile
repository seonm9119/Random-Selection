# syntax=docker/dockerfile:1.6
ARG BASE_IMAGE=pytorch/pytorch:2.12.1-cuda13.0-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RS_DATASET_DIR=/opt/random-selection/dataset \
    RS_SUPPRESS_EXTERNAL_PROGRESS=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        tini \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip wheel \
    && python -m pip install --no-cache-dir \
        numpy \
        pillow \
        pyyaml \
        tqdm

RUN mkdir -p "${RS_DATASET_DIR}" /workspace /outputs \
    && chmod -R a+rX /opt/random-selection \
    && chmod 777 /outputs

RUN python - <<'PY'
from pathlib import Path

from torchvision.datasets import CIFAR10, CIFAR100

dataset_dir = Path("/opt/random-selection/dataset")
dataset_dir.mkdir(parents=True, exist_ok=True)

for dataset_class in (CIFAR10, CIFAR100):
    for train in (True, False):
        dataset_class(root=str(dataset_dir), train=train, download=True)
PY

RUN python - <<'PY'
import numpy
import PIL
import torch
import torchvision
import tqdm
import yaml

print(f"torch={torch.__version__}")
print(f"torchvision={torchvision.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"numpy={numpy.__version__}")
print(f"pillow={PIL.__version__}")
print(f"pyyaml={yaml.__version__}")
print(f"tqdm={tqdm.__version__}")
PY

WORKDIR /workspace
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
