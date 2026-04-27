import argparse
from datetime import datetime
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
    return f"dim{m['encoder_dimension']}_blocks{m['num_blocks']}_Dt{g['Dt']}"


def build_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _store_classification_report(report_df: pd.DataFrame, run_dir: Path):
    accuracy = report_df.loc["accuracy", "f1-score"]
    report_df = report_df.drop("accuracy")
    report_df.to_csv(run_dir / "classification_report.csv")
    with open(run_dir / "classification_report.csv", "a") as f:
        f.write(f"\naccuracy,{accuracy:.2f}")

    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.axis("off")
    table = ax.table(
        cellText=report_df.values,
        rowLabels=report_df.index,
        colLabels=report_df.columns,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    ax.text(
        -0.164, 0.1,
        f"accuracy: {accuracy:.2f}",
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
        ha="left",
    )
    fig.savefig(run_dir / "classification_report.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


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
    run_id = build_run_id()

    run_dir = Path(cfg["output_dir"]) / run_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = CSVLogger(
        save_dir=cfg["output_dir"],
        name=run_name,
        version=run_id,
    )

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    data = pd.read_csv(cfg["data"]["path"], skiprows=1)
    data = TracksDataFrame(data, frame_rate=cfg["data"]["frame_rate"])

    data_description = data.describe_tracks()
    display_labels = data_description["particle_types"]

    val_tracks = cfg["data"].get("val_tracks")
    if val_tracks is not None:
        train_data, val_data = data.split_train_test_manual(val_tracks)
    else:
        train_data, val_data = data.split_train_test(
            test_size=cfg["data"]["test_size"],
            seed=cfg["seed"],
        )
        cfg["data"]["val_tracks"] = "random_split"

    class _InlineListDumper(yaml.Dumper):
        pass
    _InlineListDumper.add_representer(
        list,
        lambda dumper, data: dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True),
    )
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, Dumper=_InlineListDumper, default_flow_style=False, sort_keys=False)

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
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["val_batch_size"],
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
            lr=float(cfg["training"]["lr"]),
            weight_decay=float(cfg["training"]["weight_decay"]),
        ),
        loss=nn.CrossEntropyLoss(),
        num_classes=num_classes,
    ).build()

    # -------------------------------------------------------------------------
    # Callbacks and logger
    # -------------------------------------------------------------------------
    checkpoint_cb = ModelCheckpoint(
        dirpath=run_dir,
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    callbacks = [checkpoint_cb]

    if trial is not None:
        from optuna.integration import PyTorchLightningPruningCallback
        callbacks.append(PyTorchLightningPruningCallback(trial, monitor="val_loss"))

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

    fig, ax = trainer.history.plot()
    fig.savefig(run_dir / "training_curves.png")
    plt.close(fig)

    best_val_loss = checkpoint_cb.best_model_score.item() if checkpoint_cb.best_model_score is not None else float("inf")

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
        cm_df.to_csv(run_dir / "confusion_matrix.csv")

        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels).plot(ax=ax)
        fig.savefig(run_dir / "confusion_matrix.png", bbox_inches="tight", dpi=150)
        plt.close(fig)

        # Classification report
        report = classification_report(
            truth, preds, target_names=display_labels, output_dict=True
        )
        report_df = pd.DataFrame(report).T.round(2)
        print(report_df.to_string())
        _store_classification_report(report_df, run_dir)

    return best_val_loss, run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run(cfg)
