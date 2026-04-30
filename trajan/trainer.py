from datetime import datetime
from pathlib import Path

import numpy as np
import deeplay as dl
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from sklearn.metrics import classification_report, confusion_matrix
from torch_geometric.loader import DataLoader
from torchvision.transforms import Compose

import lightning as L
from trajan.custom_models.magik import MagikMPM
from trajan.data import TracksDataFrame
from trajan.dataset import GraphDataset
from trajan.graph import GraphFromTrajectories
from trajan.transforms import RandomFlip, RandomRotation


def check_val_coverage(dataset: "GraphDataset", num_classes: int, display_labels: list) -> None:
    """Raise ValueError if any class has no validation trajectories"""
    present = {int(g.graph_label.item()) for g in dataset.graph_dataset}
    missing = set(range(num_classes)) - present
    if missing:
        missing_names = [display_labels[i] for i in sorted(missing)]
        raise ValueError(
            f"Validation set has no trajectories for "
            f"class(es): {missing_names}. Use a different split or reduce Dt_range."
        )


def build_run_name(cfg: dict) -> str:
    g = cfg["graph"]
    return f"Dt_min{g['Dt_range'][0]}_Dt_max{g['Dt_range'][1]}"


def build_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def evaluate(model, val_loader, display_labels):
    """Run inference and return classification metrics.

    Parameters
    ----------
    model : dl.CategoricalClassifier
        Trained model to evaluate.
    val_loader : DataLoader
        Validation data loader.
    display_labels : list[str]
        Class names in the order used by the model's output logits.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        report_df : classification report with precision/recall/f1 per class.
        cm_df : confusion matrix with class names as index and columns.
    """
    model.eval()
    truth, preds = [], []
    with torch.no_grad():
        for batch in val_loader:
            y_pred = torch.argmax(model(batch), dim=1)
            truth.append(batch.y)
            preds.append(y_pred)

    truth = torch.concat(truth).numpy()
    preds = torch.concat(preds).numpy()

    report = classification_report(truth, preds, target_names=display_labels, output_dict=True)
    report_df = pd.DataFrame(report).T.round(2)

    cm = confusion_matrix(truth, preds)
    cm_df = pd.DataFrame(cm, index=display_labels, columns=display_labels)

    return report_df, cm_df


def run(cfg: dict, trial=None):
    """Train a model from a config dictionary.

    Parameters
    ----------
    cfg : dict
        Parsed YAML config dictionary.
    trial : optuna.Trial, optional
        If provided, enables Optuna pruning via a Lightning callback.

    Returns
    -------
    tuple[float, Path, dl.CategoricalClassifier, DataLoader, DataLoader, list[str]]
        best_val_loss, run_dir, best_model, train_loader, val_loader, display_labels.
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
    Dt_range = cfg["graph"]["Dt_range"]
    max_frame_distance = cfg["graph"]["max_frame_distance"]

    graph_builder = GraphFromTrajectories.from_tracks(
        train_data, max_frame_distance
    )

    train_graphs = graph_builder(train_data, target_column="type", split_tracks=True)
    val_graphs = graph_builder(val_data, target_column="type", split_tracks=True)

    # -------------------------------------------------------------------------
    # Datasets
    # -------------------------------------------------------------------------
    transform = Compose([RandomRotation(), RandomFlip()])

    dm = cfg["training"]["dataset_size_multiplier"]
    train_dataset_size = int(dm * sum(len(g.x) for g in train_graphs) / np.mean(Dt_range))
    val_dataset_size = 5 * int(dm * sum(len(g.x) for g in val_graphs) / np.mean(Dt_range))

    train_dataset = GraphDataset(
        train_graphs,
        Dt_range,
        train_dataset_size,
        transform=transform,
        target="global",
        sample_balanced=cfg["training"]["sample_balanced"],
    )
    val_dataset = GraphDataset(
        val_graphs,
        Dt_range,
        val_dataset_size,
        target="global",
    )

    check_val_coverage(val_dataset, cfg["model"]["num_classes"], display_labels)
    
    example_graph = train_dataset.__getitem__(0)
    num_extra_features = example_graph.graph_features.shape[-1]

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

    magik = MagikMPM(
        [encoder_dimension] * num_blocks,
        out_features=num_classes,
        out_activation=nn.Softmax(dim=1),
    )

    magik.head.configure(in_features=encoder_dimension+num_extra_features, out_features=num_classes)

    if cfg["training"].get("weighted_loss", False):
        labels = np.array([g.graph_label.item() for g in train_graphs])
        counts = np.bincount(labels, minlength=num_classes).astype(float)
        class_weights = torch.tensor(len(labels) / (num_classes * counts), dtype=torch.float)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = nn.CrossEntropyLoss()

    model = dl.CategoricalClassifier(
        model=magik,
        optimizer=dl.Adam(
            lr=float(cfg["training"]["lr"]),
            weight_decay=float(cfg["training"]["weight_decay"]),
        ),
        loss=loss_fn,
        num_classes=num_classes,
    ).build()

    if "scheduler" in cfg["training"]:
        import types
        _lr = float(cfg["training"]["lr"])
        _wd = float(cfg["training"]["weight_decay"])
        _warmup_epochs = int(cfg["training"]["scheduler"].get("warmup_epochs", 5))
        _start_factor = float(cfg["training"]["scheduler"].get("start_factor", 0.1))
        _eta_min = float(cfg["training"]["scheduler"].get("eta_min", 0))
        _num_epochs = int(cfg["training"]["num_epochs"])

        def _configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=_lr, weight_decay=_wd)
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=_start_factor, end_factor=1.0, total_iters=_warmup_epochs
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, _num_epochs - _warmup_epochs), eta_min=_eta_min
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[_warmup_epochs]
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

        model.configure_optimizers = types.MethodType(_configure_optimizers, model)

    # -------------------------------------------------------------------------
    # Callbacks and logger
    # -------------------------------------------------------------------------
    checkpoint_cb = ModelCheckpoint(
        dirpath=run_dir,
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

    fig, _ = trainer.history.plot()
    fig.savefig(run_dir / "training_curves.png")
    plt.close(fig)

    best_val_loss = checkpoint_cb.best_model_score.item() if checkpoint_cb.best_model_score is not None else float("inf")

    if checkpoint_cb.best_model_path:
        best_model = dl.CategoricalClassifier.load_from_checkpoint(checkpoint_cb.best_model_path)
    else:
        best_model = model

    return best_val_loss, run_dir, best_model, train_loader, val_loader, display_labels
