import deeplay as dl
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)
from torch_geometric.loader import DataLoader
from torchvision.transforms import Compose

from trajan.data import TracksDataFrame
from trajan.dataset import GraphDataset
from trajan.graph import GraphFromTrajectories
from trajan.transforms import RandomFlip, RandomRotation
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

Dt = 50
max_frame_distance = 3

#data = pd.read_csv("data/simulated/fbm_dataset.csv")
#data = particle_tracking.TracksDataFrame(data)

data = pd.read_csv("data/cytoplasmic_data/tracks.csv", skiprows=1)
data = TracksDataFrame(data, frame_rate=10)

data_description = data.describe_tracks()
display_labels = data_description['particle_types']

train_data, val_data = data.split_train_test()

graph_builder, position_scale = GraphFromTrajectories.from_tracks(train_data, Dt, max_frame_distance)

train_graphs = graph_builder(train_data, target_column="type", split_tracks=True)
val_graphs = graph_builder(val_data, target_column="type", split_tracks=True)

train_dataset_size = 2 * len(train_graphs)
val_dataset_size = 2 * len(val_graphs)

batch_size = 16
sample_balanced = True

transform = Compose([
        RandomRotation(),
        RandomFlip(),
        ])

train_dataset = GraphDataset(
    train_graphs,
    Dt,
    train_dataset_size,
    position_scale=position_scale,
    transform=transform,
    target="global",
    sample_balanced=True,
)
val_dataset = GraphDataset(
    val_graphs,
    Dt,
    train_dataset_size,
    position_scale=position_scale,
    transform=transform,
    target="global"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=0,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=256,
    drop_last=True,
    num_workers=0,
)

lr = 5e-4
wd = 1e-5
encoder_dimension = 96
num_blocks = 3
batch_size = 16
num_epochs = 10
#class_weights = torch.tensor([1.89, 5.56, 3.45])

magik = dl.GraphToGlobalMPM(
    [encoder_dimension] * num_blocks,
    out_activation=nn.Softmax(dim=1),
    out_features=3,
).create()

classifier_magik = dl.CategoricalClassifier(
    model=magik,
    optimizer=dl.Adam(lr=lr, weight_decay=wd),
    loss=nn.CrossEntropyLoss(),
    num_classes=3,
).build()

trainer_magik = dl.Trainer(max_epochs=num_epochs, accelerator="auto")

trainer_magik.fit(classifier_magik, train_loader, val_loader)

trainer_magik.history.plot()

truth, preds = [], []

for batch in val_loader:
    y_true = batch.y
    y_pred = torch.argmax(classifier_magik(batch), dim=1)
    truth.append(y_true)
    preds.append(y_pred)

truth, preds = torch.concat(truth).numpy(), torch.concat(preds).numpy()

cm = confusion_matrix(truth, preds)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                            display_labels=display_labels)

disp.plot()

report = classification_report(truth, preds, target_names=display_labels, output_dict=True)
report_df = pd.DataFrame(report).T.round(2)
report_df.to_csv("classification_report_fbm.csv", index=False)
print(report_df.to_string())

"""
hyperparams = {
    'connectivity_radius': connectivity_radius,
    'max_frame_distance': max_frame_distance,
    'Dt': Dt,
    'lr': lr,
    'wd': wd,
    'encoder_dimension': encoder_dimension,
    'num_blocks': num_blocks,
    'apply_transform': apply_transform,
    'batch_size': batch_size,
    'num_epochs': num_epochs,
    'dataset_size': int(dataset_size),
    'sample_balanced': sample_balanced,
}

with open('experiment_fbm.json', 'w') as f:
    json.dump(hyperparams, f, indent=4)

print('end')
"""
"""for Dt in [40]:
    def objective(trial):
        connectivity_radius = trial.suggest_categorical("connectivity_radius", [0.02, 0.03, 0.04, 0.05])
        max_frame_distance = trial.suggest_int("max_frame_distance", 2, 5)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        wd = trial.suggest_float("wd", 1e-6, 1e-3, log=True)
        encoder_dimension = trial.suggest_categorical("encoder_dimension", [32, 64, 96])
        num_blocks = trial.suggest_categorical("num_blocs", [2, 3, 4])

        graph_constructor = utils_magik.GraphFromTrajectories(
            connectivity_radius=connectivity_radius,
            max_frame_distance=max_frame_distance,
        )

        train_graph = graph_constructor(df=train_data, target="type", split_particles=True)
        val_graph = graph_constructor(df=val_data, target="type", split_particles=True)

        train_dataset = utils_magik.GraphDataset(
            train_graph,
            dataset_size=dataset_size,
            Dt=Dt,
            target='global',
        )
        val_dataset = utils_magik.GraphDataset(
            val_graph,
            dataset_size=dataset_size//4,
            Dt=Dt,
            target='global',
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=16,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=256,
            drop_last=True,
            num_workers=0,
        )

        magik = dl.GraphToGlobalMPM(
            [encoder_dimension,] * num_blocks,
            out_activation=torch.nn.Softmax(dim=1),
            out_features=3,
        ).create()

        classifier_magik = dl.CategoricalClassifier(
            model=magik,
            optimizer=dl.Adam(lr=lr, weight_decay=wd),
            loss=nn.CrossEntropyLoss(),
            num_classes=3,
        ).build()

        trainer_magik = dl.Trainer(max_epochs=8, accelerator="auto")

        trainer_magik.fit(classifier_magik, train_loader, val_loader)

        if trial.number % 5 == 0:
            time.sleep(200)
        return trainer_magik.callback_metrics["val_loss"].item()

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
        )
    study.optimize(objective, n_trials=40)

    with open(f"results/Magik/traj_len_{Dt}_best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=4)
"""