import argparse

import matplotlib.pyplot as plt
import yaml

from trajan.trainer import evaluate, run
from trajan.visualization import (plot_classification_report,
                                  plot_confusion_matrix,
                                  save_classification_report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    best_val_loss, run_dir, best_model, train_loader, val_loader, display_labels = run(cfg)

    for loader, split in [(train_loader, "train"), (val_loader, "val")]:
        split_dir = run_dir / split
        split_dir.mkdir()
        report_df, cm_df = evaluate(best_model, loader, display_labels)
        print(f"--- {split} ---")
        print(report_df.to_string())

        fig = plot_confusion_matrix(cm_df, display_labels)
        fig.savefig(split_dir / "confusion_matrix.png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        cm_df.to_csv(split_dir / "confusion_matrix.csv")

        fig = plot_classification_report(report_df)
        fig.savefig(split_dir / "classification_report.png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        save_classification_report(report_df, split_dir / "classification_report.csv")
