import argparse
import shutil
from pathlib import Path

import deeplay as dl
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from torch_geometric.loader import DataLoader
from torchvision.transforms import Compose

import lightning as L
from trajan.data import TracksDataFrame
from trajan.dataset import GraphDataset
from trajan.graph import GraphFromTrajectories
from trajan.transforms import RandomFlip, RandomRotation


def build_run_name(cfg: dict) -> str:
    m = cfg["model"]
    g = cfg["graph"]
    t = cfg["training"]
    return (
        f"dim{m['encoder_dimension']}"
        f"_blocks{m['num_blocks']}"
        f"_Dt{g['Dt']}"
        f"_lr{t['lr']}"
        f"_wd{t['weight_decay']}"
    )


def run(cfg: dict, trial=None) -> float:
    """Run a full training pipeline from a config dictionary.

    Parameters
    ----------
    cfg : dict
        Parsed YAML config dictionary.
    trial : optuna.Trial, optional
        If provided, enables Optuna pruning via a Lightning callback.

    Returns
    -------
    float
        Best validation loss achieved during training.
    """
    L.seed_everything(cfg["seed"])

    run_name = build_run_name(cfg)
    output_dir = Path("outputs") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config to output directory for reproducibility
    shutil.copy(args.config, output_dir / "config.yaml")

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    data = pd.read_csv(cfg["data"]["path"], skiprows=1)
    data = TracksDataFrame(data, frame_rate=cfg["data"]["frame_rate"])

    data_description = data.describe_tracks()
    display_labels = data_description["particle_types"]

    train_data, val_data = data.split_train_test(
        test_size=cfg["data"]["test_size"],
        seed=cfg["seed"],
    )

    # -------------------------------------------------------------------------
    # Graphs
    # -------------------------------------------------------------------------
    Dt = cfg["graph"]["Dt"]
    max_frame_distance = cfg["graph"]["max_frame_distance"]

    graph_builder, position_scale = GraphFromTrajectories.from_tracks(
        train_data, Dt, max_frame_distance
    )

    train_graphs = graph_builder(train_data, target_column="type", split_tracks=True)
    val_graphs = graph_builder(val_data, target_column="type", split_tracks=True)

    # -------------------------------------------------------------------------
    # Datasets
    # -------------------------------------------------------------------------
    transform = Compose([RandomRotation(), RandomFlip()])

    train_dataset_size = cfg["training"]["dataset_size_multiplier"] * len(train_graphs)
    val_dataset_size = cfg["training"]["dataset_size_multiplier"] * len(val_graphs)

    train_dataset = GraphDataset(
        train_graphs,
        Dt,
        train_dataset_size,
        position_scale=position_scale,
        transform=transform,
        target="global",
        sample_balanced=cfg["training"]["sample_balanced"],
    )
    val_dataset = GraphDataset(
        val_graphs,
        Dt,
        val_dataset_size,
        position_scale=position_scale,
        target="global",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["val_batch_size"],
        drop_last=True,
        num_workers=0,
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    num_classes = cfg["model"]["num_classes"]
    encoder_dimension = cfg["model"]["encoder_dimension"]
    num_blocks = cfg["model"]["num_blocks"]

    magik = dl.GraphToGlobalMPM(
        [encoder_dimension] * num_blocks,
        out_activation=nn.Softmax(dim=1),
        out_features=num_classes,
    ).create()

    model = dl.CategoricalClassifier(
        model=magik,
        optimizer=dl.Adam(
            lr=cfg["training"]["lr"],
            weight_decay=cfg["training"]["weight_decay"],
        ),
        loss=nn.CrossEntropyLoss(),
        num_classes=num_classes,
    ).build()

    # -------------------------------------------------------------------------
    # Callbacks and logger
    # -------------------------------------------------------------------------
    checkpoint_cb = ModelCheckpoint(
        dirpath=output_dir,
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    callbacks = [checkpoint_cb]

    if trial is not None:
        from optuna.integration import PyTorchLightningPruningCallback
        callbacks.append(PyTorchLightningPruningCallback(trial, monitor="val_loss"))

    logger = CSVLogger(save_dir="outputs", name=run_name)

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    trainer = dl.Trainer(
        max_epochs=cfg["training"]["num_epochs"],
        accelerator="auto",
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=trial is None,
    )

    trainer.fit(model, train_loader, val_loader)

    best_val_loss = checkpoint_cb.best_model_score.item()

    # -------------------------------------------------------------------------
    # Evaluation (skipped during HPO)
    # -------------------------------------------------------------------------
    if trial is None:
        best_model = dl.CategoricalClassifier.load_from_checkpoint(
            checkpoint_cb.best_model_path
        )
        best_model.eval()

        truth, preds = [], []
        with torch.no_grad():
            for batch in val_loader:
                y_pred = torch.argmax(best_model(batch), dim=1)
                truth.append(batch.y)
                preds.append(y_pred)

        truth = torch.concat(truth).numpy()
        preds = torch.concat(preds).numpy()

        # Confusion matrix
        cm = confusion_matrix(truth, preds)
        cm_df = pd.DataFrame(cm, index=display_labels, columns=display_labels)
        cm_df.to_csv(output_dir / "confusion_matrix.csv")

        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels).plot(ax=ax)
        fig.savefig(output_dir / "confusion_matrix.png", bbox_inches="tight", dpi=150)
        plt.close(fig)

        # Classification report
        report = classification_report(
            truth, preds, target_names=display_labels, output_dict=True
        )
        report_df = pd.DataFrame(report).T.round(2)
        report_df.to_csv(output_dir / "classification_report.csv")

        print(report_df.to_string())

    return best_val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run(cfg)
