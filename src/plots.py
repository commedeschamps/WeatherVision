"""Rebuild all scientific figures from stored data, metrics, and predictions."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import CLASSES, LABELS

COLORS = ["#3366A0", "#D9822B", "#3F8068", "#A45670"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.dpi": 110,
                     "savefig.dpi": 300, "pdf.fonttype": 42})


def save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(fig)


def dataset_figures(frame, root, directory):
    from .data import open_rgb
    counts = pd.crosstab(frame.label, frame.split).reindex(index=CLASSES, columns=["train", "val", "test"])
    counts.to_csv(directory / "class_distribution.csv")
    fig, ax = plt.subplots(figsize=(6.4, 3.5), layout="constrained")
    counts.plot.bar(ax=ax, color=COLORS[:3], rot=0)
    ax.set(xlabel="Visual category", ylabel="Retained images", title="Shared dataset membership")
    ax.legend(title="Split", frameon=False)
    save(fig, directory / "class_distribution.png")
    fig, axes = plt.subplots(2, 4, figsize=(9, 4.7), layout="constrained")
    for j, label in enumerate(CLASSES):
        # Training-only inspection avoids using test imagery during model design.
        subset = frame[(frame.label == label) & (frame.split == "train")].head(2)
        for i in range(2):
            axes[i, j].axis("off")
            if i < len(subset):
                row = subset.iloc[i]
                axes[i, j].imshow(open_rgb(root / row.path))
                axes[i, j].set_title(f"{label.title()} · train", fontsize=9)
    save(fig, directory / "inspection.png")


def training_curves(history, path):
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.65), layout="constrained")
    for ax, metric, label in zip(axes, ["loss", "accuracy", "f1_macro"], ["Loss", "Accuracy", "Macro-F1"]):
        for prefix, color in (("train", COLORS[0]), ("val", COLORS[1])):
            ax.plot(frame.epoch, frame[f"{prefix}_{metric}"], marker="o", markersize=3, color=color, label=prefix.title())
        ax.set(xlabel="Epoch", ylabel=label)
        if metric != "loss":
            ax.set_ylim(0, 1)
        ax.set_xticks(frame.epoch.iloc[::2] if len(frame) > 8 else frame.epoch)
    axes[0].legend(frameon=False)
    save(fig, path)


def confusion_figure(matrix, path, title="Held-out test predictions"):
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(4.6, 4.1), layout="constrained")
    im = ax.imshow(matrix, cmap="Blues", vmin=0)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() * .55 else "#182B37")
    ax.set(xticks=range(4), yticks=range(4), xticklabels=LABELS, yticklabels=LABELS,
           xlabel="Predicted category", ylabel="True category", title=title)
    fig.colorbar(im, ax=ax, shrink=.8, label="Images")
    save(fig, path)


def comparison_figure(frame, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
    for ax, metric, title in zip(axes, ["accuracy", "f1_macro"], ["Test accuracy", "Test macro-F1"]):
        bars = ax.bar(frame.run, frame[metric], color=COLORS[:len(frame)])
        ax.set(ylabel=title, ylim=(0, 1.08))
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    save(fig, path)
