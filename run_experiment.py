from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent


def load_yaml_config(config_path: Path) -> tuple[dict, str]:
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
        raise SystemExit(
            "PyYAML is required to run this infra. Install it in your experiment environment "
            "(for example: pip install pyyaml)."
        ) from exc

    raw_text = config_path.read_text(encoding="utf-8")
    return yaml.safe_load(raw_text), raw_text


def ensure_output_dir(config: dict) -> Path:
    experiment_name = config["experiment"]["name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(config["output"].get("base_dir", ROOT_DIR / "output"))
    if not base_dir.is_absolute():
        base_dir = ROOT_DIR / base_dir
    output_dir = base_dir / f"{experiment_name}_{timestamp}"
    for subdir in ("results", "plots", "logs"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    return output_dir


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ModuleNotFoundError:  # pragma: no cover - depends on env
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_environment_info() -> dict:
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
        info["gpus"] = [
            {"index": idx, "name": torch.cuda.get_device_name(idx)}
            for idx in range(torch.cuda.device_count())
        ]
    except ModuleNotFoundError:
        info["torch_version"] = None
        info["cuda_available"] = False

    try:
        import transformers

        info["transformers_version"] = transformers.__version__
    except ModuleNotFoundError:
        info["transformers_version"] = None

    return info


def collect_git_info(start_dir: Path) -> dict:
    try:
        result = subprocess.run(
            ["git", "-C", str(start_dir), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return {"available": False, "reason": "not in a git repo"}

    repo_root = Path(result.stdout.strip())
    branch = subprocess.run(
        ["git", "-C", str(repo_root), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "available": True,
        "repo_root": str(repo_root),
        "branch": branch,
        "commit": commit,
        "dirty": bool(dirty),
    }


def build_experiment(config: dict, output_dir: Path):
    experiment_type = config["experiment"]["type"]
    if experiment_type == "probing":
        from experiments.probing import ProbingExperiment

        return ProbingExperiment(config, output_dir)
    if experiment_type == "patching":
        from experiments.patching import PatchingExperiment

        return PatchingExperiment(config, output_dir)
    if experiment_type == "attention_analysis":
        from experiments.attention_analysis import AttentionAnalysisExperiment

        return AttentionAnalysisExperiment(config, output_dir)
    raise ValueError(f"Unsupported experiment type: {experiment_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a VLM interpretability experiment.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a YAML config file.")
    args = parser.parse_args()

    config, raw_config = load_yaml_config(args.config)
    output_dir = ensure_output_dir(config)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "config.yaml").write_text(raw_config, encoding="utf-8")

    seed = int(config.get("runtime", {}).get("seed", 42))
    set_global_seed(seed)

    (logs_dir / "environment.json").write_text(
        json.dumps(collect_environment_info(), indent=2),
        encoding="utf-8",
    )
    (logs_dir / "git_info.json").write_text(
        json.dumps(collect_git_info(ROOT_DIR), indent=2),
        encoding="utf-8",
    )

    experiment = build_experiment(config, output_dir)
    experiment.log(f"Starting experiment {config['experiment']['name']}")
    experiment.log(f"Config path: {args.config}")

    start = time.time()
    try:
        experiment.setup()
        experiment.run()
    except Exception as exc:
        experiment.log(f"Experiment failed: {exc}")
        raise
    finally:
        elapsed = time.time() - start
        experiment.log(f"Elapsed seconds: {elapsed:.2f}")


if __name__ == "__main__":
    main()
