import subprocess
import sys
from pathlib import Path

import torchvision


PROJECT_DIR = Path(__file__).resolve().parents[1]
PYTHON_BIN = PROJECT_DIR / ".venv" / "bin" / "python"
DATASET_DIR = PROJECT_DIR / "dataset"
READY_PATH = DATASET_DIR / ".cifar_ready"


def ensure_cifar_datasets():
    DATASET_DIR.mkdir(exist_ok=True)
    torchvision.datasets.CIFAR10(root=str(DATASET_DIR), train=True, download=True)
    torchvision.datasets.CIFAR100(root=str(DATASET_DIR), train=True, download=True)
    READY_PATH.write_text("ready\n", encoding="utf-8")


def run_search(dataset, output_path):
    command = [
        str(PYTHON_BIN),
        "-B",
        "scripts/rscl_smoke_hparam_search.py",
        "--dataset",
        dataset,
        "--batch-sizes",
        "512",
        "--warmup-epochs",
        "10",
        "--observe-epochs",
        "50",
        "--val-every-epochs",
        "5",
        "--val-steps",
        "1",
        "--num-workers",
        "0",
        "--device",
        "cuda",
        "--amp",
        "--output-path",
        output_path,
    ]
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


def main():
    ensure_cifar_datasets()
    run_search("cifar10", "results/smoke/rscl_cifar10_batch512_hparam_search_final_parallel.json")
    run_search("cifar100", "results/smoke/rscl_cifar100_batch512_hparam_search_final_parallel.json")


if __name__ == "__main__":
    sys.exit(main())
