import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import optuna
import yaml

from trajan.trainer import run


def _repo_root() -> Path:
    return Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()
    )


def objective(trial, base_cfg: dict) -> float:
    cfg = base_cfg.copy()
    cfg["model"] = base_cfg["model"].copy()
    cfg["training"] = base_cfg["training"].copy()

    cfg["model"]["encoder_dimension"] = trial.suggest_categorical("encoder_dimension", [32, 64, 96, 128])
    cfg["model"]["num_blocks"] = trial.suggest_int("num_blocks", 1, 5)
    cfg["training"]["lr"] = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    cfg["training"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)

    best_val_loss, *_ = run(cfg, trial=trial)
    return best_val_loss


if __name__ == "__main__":
    os.chdir(_repo_root())

    with open("configs/base.yaml") as f:
        base_cfg = yaml.safe_load(f)

    runs_dir = Path(base_cfg["output_dir"])
    runs_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction="minimize",
        storage=f"sqlite:///{runs_dir}/optuna.db",
        study_name="trajan_hpo",
        load_if_exists=True,
    )
    study.optimize(lambda trial: objective(trial, base_cfg), n_trials=50)

    print("\nBest params:", study.best_params)
    print(f"Best val loss: {study.best_value:.4f}")

    best_cfg = yaml.safe_load(open("configs/base.yaml"))
    best_cfg["model"]["encoder_dimension"] = study.best_params["encoder_dimension"]
    best_cfg["model"]["num_blocks"] = study.best_params["num_blocks"]
    best_cfg["training"]["lr"] = study.best_params["lr"]
    best_cfg["training"]["weight_decay"] = study.best_params["weight_decay"]

    with open("configs/best.yaml", "w") as f:
        yaml.dump(best_cfg, f, default_flow_style=False)
    print("Best config saved to configs/best.yaml")
