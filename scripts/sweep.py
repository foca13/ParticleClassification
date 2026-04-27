"""
Random recording-level sweep.

For each split, randomly assigns a fraction of recordings per class to
validation and the rest to training. Repeats for n_splits with different seeds.

Output: {output_dir}/sweep/predictions.csv
    split_id | seed | type | val_set | trajectory_idx | correct | run_dir

Aggregate per-recording accuracy downstream with:
    df.groupby(["type", "val_set"])["correct"].mean()
"""
import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import deeplay as dl
import pandas as pd
import torch
import yaml
from torch_geometric.data import Batch

from scripts.train import run
from trajan.data import TracksDataFrame
from trajan.graph import GraphFromTrajectories


def _repo_root() -> Path:
    return Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()
    )


def _generate_splits(type_sets: dict, n_splits: int, test_size: float, master_seed: int = 0) -> list[dict]:
    """Generate random recording-level train/val splits.

    For each split, independently samples `test_size` fraction of recordings
    per class as validation. Each split gets a unique seed for both the
    recording selection and the model training.
    """
    rng = np.random.default_rng(master_seed)
    splits = []
    for _ in range(n_splits):
        seed = int(rng.integers(0, 100_000))
        split_rng = np.random.default_rng(seed)
        val_tracks = {}
        for ptype, sets in type_sets.items():
            n_val = max(1, round(len(sets) * test_size))
            val_tracks[ptype] = sorted(split_rng.choice(sets, n_val, replace=False).tolist())
        splits.append({"val_tracks": val_tracks, "seed": seed})
    return splits


def _evaluate(model, val_data, graph_builder, position_scale, type_to_code) -> list[dict]:
    """Evaluate the model on each trajectory in the validation recordings.

    Builds graphs per recording individually so that (type, val_set) identity is
    preserved. The true class code is taken from context rather than graph_label,
    since single-type subsets always produce code 0 with pd.Categorical.
    """
    records = []
    model.eval()
    for particle_type in val_data["type"].unique():
        true_code = type_to_code[particle_type]
        type_data = val_data[val_data["type"] == particle_type]
        for set_id in type_data["set"].unique():
            recording = type_data[type_data["set"] == set_id]
            graphs = graph_builder(recording, target_column="type", split_tracks=True)
            with torch.no_grad():
                for traj_idx, graph in enumerate(graphs):
                    g = graph.clone()
                    g.x = (g.x - g.x.mean(dim=0)) / position_scale
                    batch = Batch.from_data_list([g])
                    pred_code = torch.argmax(model(batch), dim=1).item()
                    records.append({
                        "type": particle_type,
                        "val_set": int(set_id),
                        "trajectory_idx": traj_idx,
                        "pred_code": pred_code,
                        "correct": int(pred_code == true_code),
                    })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_splits", type=int, default=60)
    parser.add_argument("--test_size", type=float, default=0.25)
    parser.add_argument("--master_seed", type=int, default=0)
    args = parser.parse_args()

    n_splits = args.n_splits
    test_size = args.test_size
    master_seed = args.master_seed

    os.chdir(_repo_root())

    with open("configs/base.yaml") as f:
        base_cfg = yaml.safe_load(f)

    data = pd.read_csv(base_cfg["data"]["path"], skiprows=1)
    data = TracksDataFrame(data, frame_rate=base_cfg["data"]["frame_rate"])

    type_sets = {
        t: sorted(data[data["type"] == t]["set"].unique().tolist())
        for t in data["type"].unique()
    }
    type_to_code = {t: i for i, t in enumerate(sorted(type_sets))}

    splits = _generate_splits(type_sets, n_splits, test_size, master_seed)

    print(f"Total splits: {n_splits} | test_size={test_size} | master_seed={master_seed}")
    print(f"Recordings per class: { {t: len(s) for t, s in type_sets.items()} }")
    print(f"Type → code mapping: {type_to_code}\n")

    sweep_dir = Path(base_cfg["output_dir"]) / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    all_records = []

    for split_id, split in enumerate(splits):
        print(f"[{split_id + 1}/{n_splits}] seed={split['seed']} | val_tracks={split['val_tracks']}")

        cfg = copy.deepcopy(base_cfg)
        cfg["output_dir"] = str(sweep_dir)
        cfg["data"]["val_tracks"] = split["val_tracks"]
        cfg["seed"] = split["seed"]

        best_val_loss, run_dir = run(cfg)

        train_data, val_data = data.split_train_test_manual(split["val_tracks"])
        graph_builder, position_scale = GraphFromTrajectories.from_tracks(
            train_data, cfg["graph"]["Dt"], cfg["graph"]["max_frame_distance"]
        )

        best_model = dl.CategoricalClassifier.load_from_checkpoint(run_dir / "best.ckpt")

        records = _evaluate(best_model, val_data, graph_builder, position_scale, type_to_code)
        for r in records:
            r["split_id"] = split_id
            r["seed"] = split["seed"]
            r["run_dir"] = str(run_dir)
        all_records.extend(records)

        pd.DataFrame(all_records).to_csv(sweep_dir / "predictions.csv", index=False)
        print(f"    val_loss={best_val_loss:.4f} | {len(records)} trajectories evaluated\n")

    print(f"Sweep complete. Results saved to {sweep_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
