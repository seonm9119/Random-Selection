import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BATCH_SIZES = (32, 64, 128, 256, 512, 1024, 2048, 4096)
MODEL_DIRS = {
    "simclr": PROJECT_DIR / "benchmark" / "simclr",
    "rscl": PROJECT_DIR / "rscl",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Probe max one-step batch size for SSL models.")
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_DIRS.keys()), default=["simclr", "rscl"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=list(DEFAULT_BATCH_SIZES))
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--stop-on-failure", action="store_true", default=True)
    parser.add_argument("--output-path", default="results/smoke/max_batch_size.json")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-model", choices=MODEL_DIRS.keys())
    parser.add_argument("--worker-batch-size", type=int)
    return parser.parse_args()


def import_model_train_module(model_name):
    model_dir = MODEL_DIRS[model_name]
    sys.path.insert(0, str(PROJECT_DIR))
    sys.path.insert(0, str(model_dir))
    import train

    return train


def prepare_training_config(train_module, batch_size, image_size, device_name, amp):
    training_config = train_module.resolve_training_config()
    training_config["device"] = device_name
    training_config["amp"] = amp and device_name == "cuda"
    training_config["batch_size"] = batch_size
    training_config["image_size"] = image_size
    training_config["epochs"] = 1
    training_config["num_workers"] = 0
    training_config["save_every"] = 1

    if "random_negative_count" in training_config:
        training_config["random_negative_count"] = min(128, max(1, batch_size * 2 - 2))

    return training_config


class FakeDataloader:
    def __len__(self):
        return 1


def update_step_config(model_name, train_module, training_config):
    if model_name in ("simclr", "rscl"):
        train_module.update_training_step_config(training_config, FakeDataloader())


def run_simclr_like_step(model_name, training_config, device):
    if model_name == "simclr":
        from loss import SimclrNtXentLoss
        from model import SimclrModel
    else:
        from loss import RandomSelectionContrastiveLoss
        from model import RsclModel

    from optimizer import create_learning_rate_schedule, create_optimizer, set_optimizer_learning_rate

    model_class = SimclrModel if model_name == "simclr" else RsclModel
    criterion_class = SimclrNtXentLoss if model_name == "simclr" else RandomSelectionContrastiveLoss
    model = model_class(training_config).to(device)
    criterion = criterion_class(training_config).to(device)
    optimizer = create_optimizer(model, training_config)
    learning_rate_schedule = create_learning_rate_schedule(training_config)
    learning_rate = learning_rate_schedule.get_learning_rate(0)
    set_optimizer_learning_rate(optimizer, learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=training_config["amp"])

    first_images = torch.randn(training_config["batch_size"], 3, training_config["image_size"], training_config["image_size"], device=device)
    second_images = torch.randn_like(first_images)
    image_batch = torch.cat([first_images, second_images], dim=training_config["batch_concat_dim"])
    optimizer.zero_grad(set_to_none=training_config["optimizer_set_to_none"])

    with torch.cuda.amp.autocast(enabled=training_config["amp"]):
        _, projection_batch = model(image_batch)
        loss = criterion(projection_batch)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.item()


def run_worker(args):
    model_name = args.worker_model
    batch_size = args.worker_batch_size
    train_module = import_model_train_module(model_name)
    device = torch.device(args.device)
    training_config = prepare_training_config(train_module, batch_size, args.image_size, args.device, args.amp)
    update_step_config(model_name, train_module, training_config)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    started_at = time.time()

    try:
        if model_name in ("simclr", "rscl"):
            loss = run_simclr_like_step(model_name, training_config, device)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2
            reserved_memory_mb = torch.cuda.max_memory_reserved(device) / 1024 ** 2
        else:
            peak_memory_mb = 0.0
            reserved_memory_mb = 0.0

        print(json.dumps({
            "ok": True,
            "model": model_name,
            "batch_size": batch_size,
            "loss": loss,
            "peak_memory_mb": peak_memory_mb,
            "reserved_memory_mb": reserved_memory_mb,
            "elapsed_seconds": time.time() - started_at,
        }, sort_keys=True))
    except RuntimeError as error:
        error_text = str(error)

        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(json.dumps({
            "ok": False,
            "model": model_name,
            "batch_size": batch_size,
            "error_type": "cuda_oom" if "out of memory" in error_text.lower() else "runtime_error",
            "error": error_text.split("\n")[0],
            "elapsed_seconds": time.time() - started_at,
        }, sort_keys=True))


def get_device_info(device_name):
    if device_name == "cuda" and torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        return {
            "device": "cuda",
            "name": torch.cuda.get_device_name(0),
            "total_memory_mb": properties.total_memory / 1024 ** 2,
        }

    return {
        "device": "cpu",
        "name": "cpu",
        "total_memory_mb": None,
    }


def run_probe_worker(model_name, batch_size, args):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-model",
        model_name,
        "--worker-batch-size",
        str(batch_size),
        "--image-size",
        str(args.image_size),
        "--device",
        args.device,
    ]

    if args.amp:
        command.append("--amp")
    else:
        command.append("--no-amp")

    completed_process = subprocess.run(command, cwd=PROJECT_DIR, capture_output=True, text=True, timeout=600)
    output_lines = [line for line in completed_process.stdout.splitlines() if line.strip()]

    if not output_lines:
        return {
            "ok": False,
            "model": model_name,
            "batch_size": batch_size,
            "error_type": "process_error",
            "error": completed_process.stderr.strip() or "worker produced no output",
        }

    try:
        return json.loads(output_lines[-1])
    except json.JSONDecodeError:
        return {
            "ok": False,
            "model": model_name,
            "batch_size": batch_size,
            "error_type": "parse_error",
            "error": output_lines[-1],
            "stderr": completed_process.stderr.strip(),
        }


def run_probe(args):
    results = {
        "device": get_device_info(args.device),
        "amp": args.amp,
        "image_size": args.image_size,
        "batch_sizes": args.batch_sizes,
        "models": {},
    }

    for model_name in args.models:
        print(f"model={model_name}", flush=True)
        model_results = {
            "max_batch_size": None,
            "tested": {},
        }

        for batch_size in args.batch_sizes:
            worker_result = run_probe_worker(model_name, batch_size, args)
            model_results["tested"][str(batch_size)] = worker_result

            if worker_result["ok"]:
                model_results["max_batch_size"] = batch_size
                peak_memory = worker_result.get("peak_memory_mb", 0.0)
                print(f"  batch={batch_size} ok peak={peak_memory:.1f}MB loss={worker_result['loss']:.4f}", flush=True)
            else:
                print(f"  batch={batch_size} fail {worker_result.get('error_type')}: {worker_result.get('error')}", flush=True)

                if args.stop_on_failure:
                    break

        results["models"][model_name] = model_results

    output_path = PROJECT_DIR / args.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"result_path={output_path}", flush=True)
    return results


def main():
    args = parse_args()

    if args.worker:
        run_worker(args)
        return

    run_probe(args)


if __name__ == "__main__":
    main()
