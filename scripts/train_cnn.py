"""Train a CropSequenceCNN to classify particle types from video crops.

Each training sample is a full trajectory represented as a variable-length
sequence of single-channel image crops. Spatial features are extracted per
frame with a 2D CNN backbone and averaged over the temporal dimension before
classification.

Usage
-----
    python scripts/train_cnn.py --config configs/cnn.yaml
"""
import argparse
from datetime import datetime
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from sklearn.metrics import classification_report, confusion_matrix
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset

from trajan.custom_models.cnn import CropSequenceCNN, SingleCropCNN
from trajan.data import TracksDataFrame
from trajan.visualization import (plot_classification_report,
                                  plot_confusion_matrix,
                                  save_classification_report)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _normalize(imgs: torch.Tensor) -> torch.Tensor:
    """2nd–98th percentile normalisation of a crop batch to [0, 1]."""
    lo, hi = torch.quantile(imgs.flatten(), torch.tensor([0.02, 0.98]))
    return ((imgs - lo) / (hi - lo).clamp(min=1e-6)).clamp(0.0, 1.0)


class TrajectoryCropDataset(Dataset):
    """One sample = one trajectory = all its crops (variable length).

    Parameters
    ----------
    tracks : TracksDataFrame
        Must contain a ``crop_idx`` column linking rows to ``crops``.
    crops : np.ndarray of shape ``(N, H, W)`` or ``(N, 1, H, W)``
    label_map : dict[str, int]
        Maps particle type string to integer class index.
    min_length : int
        Trajectories shorter than this are excluded.
    augment : bool
        If True, apply random horizontal and vertical flips.
    """

    def __init__(
        self,
        tracks: TracksDataFrame,
        crops: np.ndarray,
        label_map: dict,
        min_length: int = 5,
        augment: bool = False,
    ) -> None:
        if "crop_idx" not in tracks.columns:
            raise ValueError(
                "tracks DataFrame has no 'crop_idx' column. "
                "Load an npz built with build_video_dataset.py."
            )

        crops_t = torch.tensor(np.asarray(crops, dtype=np.float32))
        if crops_t.ndim == 3:
            crops_t = crops_t.unsqueeze(1)  # (N, 1, H, W)
        self.crops = crops_t
        self.augment = augment

        self.samples: list[tuple[np.ndarray, int]] = []
        for (ptype, *_), grp in tracks.groupby(["type", "set", "label"]):
            idxs = grp.sort_values("frame")["crop_idx"].values
            if len(idxs) < min_length:
                continue
            self.samples.append((idxs, label_map[ptype]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        crop_idxs, label = self.samples[idx]
        imgs = self.crops[crop_idxs]          # (T, 1, H, W)
        imgs = _normalize(imgs)
        if self.augment:
            if torch.rand(1).item() > 0.5:
                imgs = imgs.flip(-1)           # horizontal flip
            if torch.rand(1).item() > 0.5:
                imgs = imgs.flip(-2)           # vertical flip
            angle = float(torch.rand(1).item() * 360.0 - 180.0)
            imgs = TF.rotate(imgs, angle)      # same angle applied to all T frames
        return imgs, label


def _collate(batch):
    """Concatenate variable-length sequences and build batch_idx."""
    imgs_list, labels = zip(*batch)
    batch_idx = torch.cat([
        torch.full((len(imgs),), i, dtype=torch.long)
        for i, imgs in enumerate(imgs_list)
    ])
    imgs = torch.cat(imgs_list, dim=0)         # (total_frames, 1, H, W)
    labels = torch.tensor(labels, dtype=torch.long)
    return imgs, batch_idx, labels


class CropDataset(Dataset):
    """One sample = one detection crop, normalised per-crop.

    Parameters
    ----------
    tracks : TracksDataFrame
        Must contain a ``crop_idx`` column.
    crops : np.ndarray of shape ``(N, H, W)`` or ``(N, 1, H, W)``
    label_map : dict[str, int]
    augment : bool
    """

    def __init__(
        self,
        tracks: TracksDataFrame,
        crops: np.ndarray,
        label_map: dict,
        augment: bool = False,
    ) -> None:
        crops_t = torch.tensor(np.asarray(crops, dtype=np.float32))
        if crops_t.ndim == 3:
            crops_t = crops_t.unsqueeze(1)   # (N, 1, H, W)
        self.crops = crops_t
        self.augment = augment

        valid = tracks[tracks["type"].isin(label_map)]
        self.crop_idxs = valid["crop_idx"].values.astype(int)
        self.labels = np.array([label_map[t] for t in valid["type"].values], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = self.crops[self.crop_idxs[idx]]   # (1, H, W)
        img = _normalize(img)
        label = int(self.labels[idx])
        if self.augment:
            if torch.rand(1).item() > 0.5:
                img = img.flip(-1)
            if torch.rand(1).item() > 0.5:
                img = img.flip(-2)
            angle = float(torch.rand(1).item() * 360.0 - 180.0)
            img = TF.rotate(img, angle)
        return img, label


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------

class CNNClassifier(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: torch.Tensor = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        self.save_hyperparameters(ignore=["model", "class_weights"])

    def forward(self, imgs: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        return self.model(imgs, batch_idx)

    def _step(self, batch):
        imgs, batch_idx, labels = batch
        logits = self(imgs, batch_idx)
        loss = self.loss_fn(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        return loss, acc, labels.size(0)

    def training_step(self, batch, _):
        loss, acc, bs = self._step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log("train_acc",  acc,  on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        return loss

    def validation_step(self, batch, _):
        loss, acc, bs = self._step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log("val_acc",  acc,  on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


class CropCNNClassifier(L.LightningModule):
    """Lightning wrapper for SingleCropCNN (per-crop, no batch_idx)."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: torch.Tensor = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        self.save_hyperparameters(ignore=["model", "class_weights"])

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.model(imgs)

    def _step(self, batch):
        imgs, labels = batch
        logits = self(imgs)
        loss = self.loss_fn(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        return loss, acc, labels.size(0)

    def training_step(self, batch, _):
        loss, acc, bs = self._step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log("train_acc",  acc,  on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        return loss

    def validation_step(self, batch, _):
        loss, acc, bs = self._step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log("val_acc",  acc,  on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(classifier: CNNClassifier, loader: DataLoader, display_labels: list):
    classifier.eval()
    truth_all, preds_all = [], []
    with torch.no_grad():
        for imgs, batch_idx, labels in loader:
            logits = classifier(imgs, batch_idx)
            preds_all.append(logits.argmax(dim=1))
            truth_all.append(labels)
    truth = torch.cat(truth_all).numpy()
    preds = torch.cat(preds_all).numpy()
    report = classification_report(truth, preds, target_names=display_labels, output_dict=True)
    report_df = pd.DataFrame(report).T.round(2)
    cm = confusion_matrix(truth, preds)
    cm_df = pd.DataFrame(cm, index=display_labels, columns=display_labels)
    return report_df, cm_df


def evaluate_crops(classifier: CropCNNClassifier, loader: DataLoader, display_labels: list):
    classifier.eval()
    truth_all, preds_all = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            preds_all.append(classifier(imgs).argmax(dim=1))
            truth_all.append(labels)
    truth = torch.cat(truth_all).numpy()
    preds = torch.cat(preds_all).numpy()
    report = classification_report(truth, preds, target_names=display_labels, output_dict=True)
    report_df = pd.DataFrame(report).T.round(2)
    cm = confusion_matrix(truth, preds)
    cm_df = pd.DataFrame(cm, index=display_labels, columns=display_labels)
    return report_df, cm_df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_split_summary(train_data, val_data) -> None:
    train_desc = train_data.describe_tracks(print_msg=False)
    val_desc   = val_data.describe_tracks(print_msg=False)
    classes = train_desc["particle_types"]

    col_w = max(len(c) for c in classes) + 2
    header = (
        f"  {'Class':<{col_w}}  {'Train recs':>10}  {'Train tracks':>12}"
        f"  {'Avg len':>7}  {'Val recs':>9}  {'Val tracks':>10}  {'Avg len':>7}"
    )
    print("\nTrain / Val split:")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for cls in classes:
        tr = train_desc.get(cls, {})
        va = val_desc.get(cls, {})
        print(
            f"  {cls:<{col_w}}  {tr.get('n_recordings', 0):>10}  "
            f"{tr.get('n_tracks', 0):>12}  {tr.get('avg_track_len', 0):>7.1f}"
            f"  {va.get('n_recordings', 0):>9}  "
            f"{va.get('n_tracks', 0):>10}  {va.get('avg_track_len', 0):>7.1f}"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(cfg: dict) -> None:
    L.seed_everything(cfg.get("seed", 42))

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data, crops = TracksDataFrame.load_npz(cfg["data"]["video_path"])
    if crops is None:
        raise ValueError("The npz file contains no crops.")

    data.frame_rate = cfg["data"].get("frame_rate", None)
    desc = data.describe_tracks(print_msg=False)
    display_labels = desc["particle_types"]
    label_map = {name: i for i, name in enumerate(display_labels)}

    val_tracks_cfg = cfg["data"].get("val_tracks")
    if val_tracks_cfg:
        train_data, val_data = data.split_train_test_manual(val_tracks_cfg)
    else:
        train_data, val_data = data.split_train_test(
            test_size=cfg["data"].get("test_size", 0.25),
            seed=cfg.get("seed", 42),
        )

    print_split_summary(train_data, val_data)

    mode = cfg["data"].get("mode", "trajectory")
    cnn_cfg = cfg.get("model", {})
    batch_size = cfg["training"].get("batch_size", 32)
    lr = float(cfg["training"].get("lr", 1e-3))
    weight_decay = float(cfg["training"].get("weight_decay", 1e-4))

    if mode == "crop":
        train_ds = CropDataset(train_data, crops, label_map, augment=True)
        val_ds   = CropDataset(val_data,   crops, label_map, augment=False)
        print(f"\nDataset sizes — train: {len(train_ds)}, val: {len(val_ds)}")

        if cfg["training"].get("weighted_loss", False):
            counts = np.bincount(train_ds.labels, minlength=len(display_labels)).astype(float)
            class_weights = torch.tensor(
                len(train_ds.labels) / (len(display_labels) * counts), dtype=torch.float
            )
            print(f"Class weights: { {k: round(float(v), 3) for k, v in zip(display_labels, class_weights)} }")
        else:
            class_weights = None

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

        inner_model = SingleCropCNN(
            channels=tuple(cnn_cfg.get("channels", [16, 32, 64])),
            embed_dim=cnn_cfg.get("embed_dim", 128),
            num_classes=len(display_labels),
            dropout=cnn_cfg.get("dropout", 0.3),
        )
        classifier = CropCNNClassifier(
            model=inner_model, lr=lr, weight_decay=weight_decay, class_weights=class_weights
        )

    else:  # trajectory
        min_length = cfg["data"].get("min_length", 5)
        train_ds = TrajectoryCropDataset(train_data, crops, label_map, min_length=min_length, augment=True)
        val_ds   = TrajectoryCropDataset(val_data,   crops, label_map, min_length=min_length, augment=False)
        print(f"\nDataset sizes — train: {len(train_ds)}, val: {len(val_ds)}")

        if cfg["training"].get("weighted_loss", False):
            train_labels = [label for _, label in train_ds.samples]
            counts = np.bincount(train_labels, minlength=len(display_labels)).astype(float)
            class_weights = torch.tensor(
                len(train_labels) / (len(display_labels) * counts), dtype=torch.float
            )
            print(f"Class weights: { {k: round(float(v), 3) for k, v in zip(display_labels, class_weights)} }")
        else:
            class_weights = None

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=_collate)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=_collate)

        inner_model = CropSequenceCNN(
            channels=tuple(cnn_cfg.get("channels", [16, 32, 64])),
            embed_dim=cnn_cfg.get("embed_dim", 128),
            num_classes=len(display_labels),
            dropout=cnn_cfg.get("dropout", 0.3),
        )
        classifier = CNNClassifier(
            model=inner_model, lr=lr, weight_decay=weight_decay, class_weights=class_weights
        )

    # ------------------------------------------------------------------
    # Run directory & callbacks
    # ------------------------------------------------------------------
    run_dir = Path(cfg.get("output_dir", "runs/cnn")) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = CSVLogger(save_dir=str(run_dir.parent), name="", version=run_dir.name)
    checkpoint_cb = ModelCheckpoint(dirpath=run_dir, monitor="val_loss", mode="min", save_top_k=1)
    early_stop_cb = EarlyStopping(
        monitor="val_loss", mode="min",
        patience=int(cfg["training"].get("patience", 10)),
        min_delta=float(cfg["training"].get("min_delta", 1e-4)),
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    trainer = L.Trainer(
        max_epochs=cfg["training"].get("num_epochs", 50),
        accelerator="auto",
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=logger,
    )
    trainer.fit(classifier, train_loader, val_loader)

    cfg["training"]["actual_epochs"] = trainer.current_epoch + 1

    class _InlineListDumper(yaml.Dumper):
        pass
    _InlineListDumper.add_representer(
        list,
        lambda dumper, data: dumper.represent_sequence(
            "tag:yaml.org,2002:seq", data, flow_style=True
        ),
    )
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, Dumper=_InlineListDumper, default_flow_style=False, sort_keys=False)

    # ------------------------------------------------------------------
    # Load best checkpoint and evaluate
    # ------------------------------------------------------------------
    LitClass = CropCNNClassifier if mode == "crop" else CNNClassifier
    best = (
        LitClass.load_from_checkpoint(
            checkpoint_cb.best_model_path, model=inner_model, class_weights=class_weights
        )
        if checkpoint_cb.best_model_path
        else classifier
    )
    eval_fn = evaluate_crops if mode == "crop" else evaluate

    for loader, split in [(train_loader, "train"), (val_loader, "val")]:
        split_dir = run_dir / split
        split_dir.mkdir(exist_ok=True)
        report_df, cm_df = eval_fn(best, loader, display_labels)
        print(f"\n--- {split} ---")
        print(report_df.to_string())

        fig = plot_confusion_matrix(cm_df, display_labels)
        fig.savefig(split_dir / "confusion_matrix.png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        cm_df.to_csv(split_dir / "confusion_matrix.csv")

        fig = plot_classification_report(report_df)
        fig.savefig(split_dir / "classification_report.png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        save_classification_report(report_df, split_dir / "classification_report.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cnn.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run(cfg)
