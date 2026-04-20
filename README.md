# trajan

Graph neural network-based classification of particle types from single-particle tracking data.

Particles are represented as spatiotemporal graphs, where nodes correspond to detections and edges to candidate temporal links. A GNN is trained to classify particle motion types from trajectory data.

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/<your-username>/trajan.git
cd trajan
pip install -e .
```

For notebook support:

```bash
pip install -e ".[notebooks]"
```

> **Note:** Always run scripts from the repository root.

## Requirements

- Python ≥ 3.10
- PyTorch 2.2.2 (CPU, macOS x86_64)
- See `pyproject.toml` for the full dependency list

## Data

Place your data in the `data/` directory (not tracked by git). The expected input format is `.xml` files following the tracking XML schema parsed by `trajan/io.py`. See `data/README.md` for details.

## Project structure

```
trajan/
├── trajan/
│   ├── io.py               # XML loading and parsing
│   ├── data.py             # TracksDataFrame
│   ├── graph.py            # GraphFromTrajectories
│   ├── dataset.py          # GraphDataset
│   ├── transforms.py       # Data augmentation
│   ├── loss.py             # Custom loss functions
│   ├── visualization.py    # Plotting utilities
│   └── utils.py            # Miscellaneous helpers
├── notebooks/              # Exploratory notebooks
├── scripts/                # Training and evaluation entrypoints
├── configs/                # YAML configuration files
├── tests/                  # Unit tests
└── data/                   # Local data (gitignored)
```

## Usage

```bash
python scripts/train.py --config configs/default.yaml
```

## License

MIT
