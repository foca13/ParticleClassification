from datetime import datetime
from pathlib import Path

import numpy as np
import deeplay as dl
import lightning as L
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from sklearn.metrics import classification_report, confusion_matrix
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torchvision.transforms import Compose

from trajan.custom_models.magik import MagikMPM, ImageGraphConv
from trajan.data import TracksDataFrame
from trajan.dataset import VelocityGraphDataset, PositionGraphDataset
from trajan.graph import VelocityGraphFromTrajectories, PositionGraphFromTrajectories
from trajan.transforms import RandomFlip, RandomRotation


def _collate_video(batch):
    graphs, imgs_list = zip(*batch)
    return Batch.from_data_list(list(graphs)), torch.cat(imgs_list, dim=0)


class VideoGraphClassifier(L.LightningModule):
    """LightningModule wrapping ImageGraphConv for (graph, imgs) batches."""

    def __init__(self, model, lr=1e-4, wd=1e-5, num_classes=3):
        super().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr
        self.wd = wd
        self.num_classes = num_classes
        self._val_losses: list = []

    def forward(self, graph, imgs):
        return self.model(graph, imgs)

    def _step(self, batch, stage):
        graph, imgs = batch
        y_hat = self(graph, imgs)
        loss = self.loss_fn(y_hat, graph.y)
        self.log(f"{stage}_loss", loss, batch_size=graph.num_graphs, prog_bar=True)
        if stage == "val":
            self._val_losses.append(loss.detach())
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)

    @property
    def best_val_loss(self):
        return float(torch.stack(self._val_losses).mean()) if self._val_losses else float("inf")


def check_val_coverage(dataset, num_classes: int, display_labels: list) -> None:
    """Raise ValueError if any class has no validation trajectories"""
    present = {int(g.graph_label.item()) for g in dataset.graph_dataset}
    missing = set(range(num_classes)) - present
    if missing:
        missing_names = [display_labels[i] for i in sorted(missing)]
        raise ValueError(
            f"Validation set has no trajectories for "
            f"class(es): {missing_names}. Use a different split or reduce Dt_range."
        )


def print_split_summary(train_data, val_data) -> None:
    """Print a compact per-class summary of the train/val split."""
    train_desc = train_data.describe_tracks(print_msg=False)
    val_desc = val_data.describe_tracks(print_msg=False)
    classes = train_desc["particle_types"]

    col_w = max(len(c) for c in classes) + 2
    header = f"  {'Class':<{col_w}}  {'Train recs':>10}  {'Train tracks':>12}  {'Val recs':>9}  {'Val tracks':>10}"
    print("\nTrain / Val split:")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for cls in classes:
        tr = train_desc.get(cls, {})
        va = val_desc.get(cls, {})
        print(
            f"  {cls:<{col_w}}  {tr.get('n_recordings', 0):>10}  "
            f"{tr.get('n_tracks', 0):>12}  {va.get('n_recordings', 0):>9}  "
            f"{va.get('n_tracks', 0):>10}"
        )
    print()


def build_run_name(cfg: dict) -> str:
    g = cfg["graph"]
    return f"Dt_min{g['Dt_range'][0]}_Dt_max{g['Dt_range'][1]}"


def build_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def evaluate(model, val_loader, display_labels, mode="graph"):
    """Run inference and return classification metrics.

    Parameters
    ----------
    model : dl.CategoricalClassifier or VideoGraphClassifier
        Trained model to evaluate.
    val_loader : DataLoader
        Validation data loader.
    display_labels : list[str]
        Class names in the order used by the model's output logits.
    mode : str
        ``"graph"`` for trajectory-only model, ``"video"`` for image+graph model.

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
            if mode == "video":
                graph, imgs = batch
                y_pred = torch.argmax(model(graph, imgs), dim=1)
                truth.append(graph.y)
            else:
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
    frame_rate = cfg["data"]["frame_rate"]
    training_mode = cfg["data"].get("mode", "graph")

    if training_mode == "video":
        data, crops = TracksDataFrame.load_npz(cfg["data"]["video_path"])
        data.frame_rate = frame_rate
    else:
        data = pd.read_csv(cfg["data"]["path"], skiprows=1)
        data = TracksDataFrame(data, frame_rate=frame_rate)

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

    print_split_summary(train_data, val_data)

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
    max_frame_gap = cfg["graph"]["max_frame_gap"]
    node_features = cfg["graph"].get("node_features", "velocity")

    if node_features == "position":
        graph_builder = PositionGraphFromTrajectories.from_tracks(
            train_data, max_frame_gap
        )
    else:
        graph_builder, velocity_std = VelocityGraphFromTrajectories.from_tracks(
            train_data, max_frame_gap, frame_rate=frame_rate
        )

    train_graphs = graph_builder(train_data, target_column="type", split_tracks=True)
    val_graphs = graph_builder(val_data, target_column="type", split_tracks=True)

    # -------------------------------------------------------------------------
    # Datasets
    # -------------------------------------------------------------------------
    _rot, _flip = RandomRotation(), RandomFlip()
    if training_mode == "video":
        # Compose doesn't forward the imgs arg; chain transforms manually so
        # graph and crops are rotated/flipped with the same random parameters.
        def transform(graph, imgs=None):
            graph, imgs = _rot(graph, imgs)
            graph, imgs = _flip(graph, imgs)
            return graph, imgs
    else:
        transform = Compose([_rot, _flip])

    dm = cfg["training"]["dataset_size_multiplier"]
    train_dataset_size = int(dm * sum(len(g.x) for g in train_graphs) / np.mean(Dt_range))
    val_dataset_size = 5 * int(dm * sum(len(g.x) for g in val_graphs) / np.mean(Dt_range))

    crops_arg = crops if training_mode == "video" else None

    if node_features == "position":
        train_dataset = PositionGraphDataset(
            train_graphs, Dt_range, train_dataset_size,
            transform=transform, target="global",
            sample_balanced=cfg["training"]["sample_balanced"],
            crops=crops_arg,
        )
        val_dataset = PositionGraphDataset(
            val_graphs, Dt_range, val_dataset_size,
            target="global", crops=crops_arg,
        )
    else:
        train_dataset = VelocityGraphDataset(
            train_graphs, Dt_range, train_dataset_size,
            velocity_std=velocity_std, transform=transform,
            target="global", sample_balanced=cfg["training"]["sample_balanced"],
            crops=crops_arg,
        )
        val_dataset = VelocityGraphDataset(
            val_graphs, Dt_range, val_dataset_size,
            velocity_std=velocity_std, target="global", crops=crops_arg,
        )

    check_val_coverage(val_dataset, cfg["model"]["num_classes"], display_labels)

    example_graph = train_dataset[0]
    if training_mode == "video":
        example_graph = example_graph[0]  # (graph, imgs) tuple
    num_extra_features = example_graph.graph_features.shape[-1]

    if training_mode == "video":
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg["training"]["batch_size"],
            shuffle=True, collate_fn=_collate_video,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=cfg["training"]["val_batch_size"],
            collate_fn=_collate_video,
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=cfg["training"]["batch_size"],
            shuffle=True, num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=cfg["training"]["val_batch_size"],
            num_workers=0,
        )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    num_classes = cfg["model"]["num_classes"]
    encoder_dimension = cfg["model"]["encoder_dimension"]
    num_blocks = cfg["model"]["num_blocks"]

    if training_mode == "video":
        cnn_cfg = cfg["model"].get("cnn", {})
        image_graph_conv = ImageGraphConv(
            [encoder_dimension] * num_blocks,
            out_features=num_classes,
            out_activation=nn.Softmax(dim=1),
            cnn_channels=cnn_cfg.get("channels", [8, 16, 32]),
            kernel_size=cnn_cfg.get("kernel_size", 3),
            fusion=cnn_cfg.get("fusion", "add"),
        )
        image_graph_conv.head.configure(in_features=encoder_dimension + num_extra_features, out_features=num_classes)
        image_graph_conv = image_graph_conv.create()

        model = VideoGraphClassifier(
            image_graph_conv,
            lr=float(cfg["training"]["lr"]),
            wd=float(cfg["training"]["weight_decay"]),
            num_classes=num_classes,
        )
    else:
        magik = MagikMPM(
            [encoder_dimension] * num_blocks,
            out_features=num_classes,
            out_activation=nn.Softmax(dim=1),
        )
        magik.head.configure(in_features=encoder_dimension + num_extra_features, out_features=num_classes)

        if cfg["training"].get("weighted_loss", False):
            labels_arr = np.array([g.graph_label.item() for g in train_graphs])
            counts = np.bincount(labels_arr, minlength=num_classes).astype(float)
            class_weights = torch.tensor(len(labels_arr) / (num_classes * counts), dtype=torch.float)
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
    if training_mode == "video":
        trainer = L.Trainer(
            max_epochs=cfg["training"]["num_epochs"],
            accelerator="auto",
            callbacks=callbacks,
            logger=logger,
            enable_progress_bar=trial is None,
        )
        trainer.fit(model, train_loader, val_loader)
        best_val_loss = checkpoint_cb.best_model_score.item() if checkpoint_cb.best_model_score is not None else float("inf")
        if checkpoint_cb.best_model_path:
            ckpt = torch.load(checkpoint_cb.best_model_path, map_location="cpu")
            model.load_state_dict(ckpt["state_dict"])
        best_model = model
    else:
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
