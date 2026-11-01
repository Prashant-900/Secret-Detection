"""Headless, portable plots for the reporting bundle.

This module uses only the standard library and matplotlib. It never loads model
weights, starts GPU inference, or changes an operating threshold.
"""
from __future__ import annotations

import math
from pathlib import Path
import re
import textwrap


# Okabe-Ito colors remain distinct for most forms of color-vision deficiency.
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#332288", "#666666")


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _wrap(value, width=34):
    return "\n".join(textwrap.wrap(str(value), width=width, break_long_words=True, break_on_hyphens=False))


def _aligned(labels, series):
    length = len(labels)
    for name, values in series.items():
        if len(values) != length:
            raise ValueError(f"Plot series {name!r} has {len(values)} values for {length} labels")


class PlotWriter:
    """Write PNG/SVG pairs and retain a report-ready chart inventory."""

    def __init__(self, out_dir):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError(
                "Plotting requires matplotlib. From the project root run: "
                ".venv-train/bin/python -m pip install -r reporting_bundle/requirements.txt"
            ) from exc
        self.plt = plt
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.items = []
        self.plt.rcParams.update({
            "font.family": "DejaVu Sans", "font.size": 10,
            "axes.titlesize": 14, "axes.titleweight": "bold",
            "axes.labelsize": 11, "axes.spines.top": False,
            "axes.spines.right": False, "axes.edgecolor": "#596579",
            "text.color": "#18283D", "axes.labelcolor": "#18283D",
            "xtick.color": "#42536B", "ytick.color": "#42536B",
            "figure.facecolor": "white", "axes.facecolor": "white",
            "savefig.facecolor": "white", "svg.fonttype": "none",
            "legend.frameon": False,
        })

    def _figure(self, title, width=10.0, height=5.8):
        fig, ax = self.plt.subplots(figsize=(width, height), layout="constrained")
        ax.set_title(_wrap(title, 85), loc="left", pad=18)
        ax.set_axisbelow(True)
        return fig, ax

    def _save(self, fig, name, title, caption):
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip(".")
        if not stem:
            raise ValueError("Plot name must contain a filename character")
        if stem.endswith((".png", ".svg")):
            stem = stem.rsplit(".", 1)[0]
        try:
            fig.savefig(self.out_dir / (stem + ".png"), dpi=180, bbox_inches="tight", pad_inches=0.22)
            fig.savefig(self.out_dir / (stem + ".svg"), bbox_inches="tight", pad_inches=0.22)
        finally:
            self.plt.close(fig)
        self.items.append({"filename": stem + ".png", "title": title, "caption": caption})

    @staticmethod
    def _empty(ax, message="No applicable observations"):
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color="#596579")

    @staticmethod
    def _legend(fig, ax, columns=3):
        # Matplotlib >=3.8 reserves a real layout row for this legend, so long
        # tick labels cannot collide with a legend placed below the axes.
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, [_wrap(label, 42) for label in labels], loc="outside lower center",
                   ncol=min(columns, max(1, len(labels))), fontsize=9)

    def grouped(self, name, title, labels, series, ylabel="Score", caption="", bounded=False):
        labels = list(labels)
        series = {key: list(values) for key, values in series.items()}
        _aligned(labels, series)
        width = min(22.0, max(8.6, len(labels) * 0.70))
        fig, ax = self._figure(title, width=width, height=6.2)
        groups = max(1, len(series))
        bar_width = 0.78 / groups
        drawn = False
        for j, (key, values) in enumerate(series.items()):
            usable = [(i, value) for i, value in enumerate(values) if _finite(value)]
            if not usable:
                continue
            drawn = True
            ax.bar([i - 0.39 + bar_width * (j + 0.5) for i, _ in usable],
                   [value for _, value in usable], width=bar_width * 0.92,
                   label=str(key), color=COLORS[j % len(COLORS)], zorder=3)
        ax.set_xticks(list(range(len(labels))), [_wrap(label, 24) for label in labels])
        if len(labels) > 8:
            self.plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#E0E6EE", linewidth=0.8)
        if bounded:
            ax.set_ylim(0, 1.05)
        elif drawn:
            if all(value >= 0 for values in series.values() for value in values if _finite(value)):
                ax.set_ylim(bottom=0)
            else:
                ax.axhline(0, color="#7E8B9C", linewidth=1)
                ax.margins(y=0.12)
        if drawn:
            self._legend(fig, ax, columns=4)
        else:
            self._empty(ax)
        self._save(fig, name, title, caption)

    def horizontal(self, name, title, labels, series, xlabel="Score", caption="", bounded=False):
        labels = list(labels)
        series = {key: list(values) for key, values in series.items()}
        _aligned(labels, series)
        groups = max(1, len(series))
        wrapped = [_wrap(label, 44) for label in labels]
        line_count = sum(max(1, label.count("\n") + 1) for label in wrapped)
        height = max(4.5, 2.2 + max(len(labels) * min(0.55 + 0.13 * groups, 1.3), line_count * 0.36))
        fig, ax = self._figure(title, width=11.5, height=height)
        bar_height = 0.78 / groups
        drawn = False
        for j, (key, values) in enumerate(series.items()):
            usable = [(i, value) for i, value in enumerate(values) if _finite(value)]
            if not usable:
                continue
            drawn = True
            bars = ax.barh([i - 0.39 + bar_height * (j + 0.5) for i, _ in usable],
                           [value for _, value in usable], height=bar_height * 0.92,
                           label=str(key), color=COLORS[j % len(COLORS)], zorder=3)
            if len(labels) * groups <= 60:
                for bar, (_, value) in zip(bars, usable):
                    label = f"{value:.3f}" if bounded else f"{value:,.3g}"
                    ax.annotate(label, (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                                xytext=((4 if value >= 0 else -4), 0), textcoords="offset points",
                                ha="left" if value >= 0 else "right", va="center", fontsize=8)
        ax.set_yticks(list(range(len(labels))), wrapped)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color="#E0E6EE", linewidth=0.8)
        if bounded:
            ax.set_xlim(0, 1.15)
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        elif drawn:
            if all(value >= 0 for values in series.values() for value in values if _finite(value)):
                ax.set_xlim(left=0)
            else:
                ax.axvline(0, color="#7E8B9C", linewidth=1)
            ax.margins(x=0.18)
        if drawn:
            self._legend(fig, ax, columns=4)
        else:
            self._empty(ax)
        self._save(fig, name, title, caption)

    def lines(self, name, title, x, series, xlabel="", ylabel="", caption="", bounded=False):
        x = list(x)
        series = {key: list(values) for key, values in series.items()}
        _aligned(x, series)
        fig, ax = self._figure(title)
        drawn = False
        for j, (key, values) in enumerate(series.items()):
            # NaN preserves visible gaps instead of joining across missing epochs.
            usable = [float(value) if _finite(value) else math.nan for value in values]
            if not any(math.isfinite(value) for value in usable):
                continue
            drawn = True
            ax.plot(x, usable, label=str(key), color=COLORS[j % len(COLORS)],
                    linewidth=2, marker="o" if len(x) <= 30 else None, markersize=4,
                    linestyle=("-", "--", "-.", ":")[j // len(COLORS) % 4])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(color="#E0E6EE", linewidth=0.8)
        if bounded:
            ax.set_ylim(0, 1.05)
        if x and all(type(value) is int for value in x) and len(x) <= 20:
            ax.set_xticks(x)
        if drawn:
            self._legend(fig, ax)
        else:
            self._empty(ax)
        self._save(fig, name, title, caption)

    def curve(self, name, title, x, y, xlabel, ylabel, caption="", diagonal=False):
        x, y = list(x), list(y)
        if len(x) != len(y):
            raise ValueError("Curve coordinates have different lengths")
        points = [(a, b) for a, b in zip(x, y) if _finite(a) and _finite(b)]
        fig, ax = self._figure(title, width=7.6, height=6.4)
        if diagonal:
            ax.plot([0, 1], [0, 1], linestyle="--", color="#9CA7B5", label="Chance reference")
        if points:
            # With increasing recall, AP uses each new precision value over
            # the preceding recall interval (a left-constant visual step).
            style = "steps-pre" if xlabel.lower() == "recall" and ylabel.lower() == "precision" else "default"
            ax.plot([a for a, _ in points], [b for _, b in points], color=COLORS[0], linewidth=2.2, drawstyle=style)
        else:
            self._empty(ax, "No applicable curve observations")
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1.02))
        ax.grid(color="#E0E6EE", linewidth=0.8)
        if diagonal:
            ax.legend(loc="lower right")
        self._save(fig, name, title, caption)

    def confusion(self, name, title, tn, fp, fn, tp, caption=""):
        values = [[tn, fp], [fn, tp]]
        if any(not _finite(value) or value < 0 for row in values for value in row):
            raise ValueError("Confusion counts must be finite nonnegative numbers")
        fig, ax = self._figure(title, width=7.6, height=6.5)
        maximum = max(tn, fp, fn, tp, 1)
        im = ax.imshow(values, cmap="Blues", vmin=0, vmax=maximum, aspect="equal")
        labels = [["TN", "FP"], ["FN", "TP"]]
        for row in range(2):
            support = sum(values[row])
            for col in range(2):
                value = values[row][col]
                rate = f"{100 * value / support:.1f}% of row" if support else "No row support"
                ax.text(col, row, f"{labels[row][col]} = {value:,.0f}\n{rate}", ha="center", va="center",
                        color="white" if value > maximum * 0.58 else "#18283D", fontsize=13)
        ax.set_xticks([0, 1], ["Predicted negative", "Predicted positive"])
        ax.set_yticks([0, 1], ["Explicit negative", "Explicit positive"])
        ax.set_xlabel("Model decision at the frozen threshold")
        ax.set_ylabel("Candidate annotation")
        ax.spines[:].set_visible(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05, label="Candidate count")
        self._save(fig, name, title, caption)

    def reliability(self, name, title, bins, caption=""):
        rows = [row for row in bins if row.get("count", 0) > 0
                and _finite(row.get("mean_score")) and _finite(row.get("positive_rate"))]
        fig, ax = self._figure(title, width=7.6, height=6.4)
        ax.plot([0, 1], [0, 1], linestyle="--", color="#8B97A8", label="Perfect reliability")
        if rows:
            rows.sort(key=lambda row: row["mean_score"])
            largest = max(row["count"] for row in rows)
            ax.plot([row["mean_score"] for row in rows], [row["positive_rate"] for row in rows],
                    color=COLORS[0], linewidth=1.7)
            ax.scatter([row["mean_score"] for row in rows], [row["positive_rate"] for row in rows],
                       s=[30 + 270 * math.sqrt(row["count"] / largest) for row in rows],
                       color=COLORS[0], edgecolors="white", linewidth=0.7, zorder=4,
                       label="Observed bins (size reflects count)")
        else:
            self._empty(ax)
        ax.set(xlabel="Mean predicted secret probability", ylabel="Observed explicit-positive fraction",
               xlim=(0, 1), ylim=(0, 1))
        ax.grid(color="#E0E6EE", linewidth=0.8)
        self._legend(fig, ax, columns=1)
        self._save(fig, name, title, caption)

    def histogram(self, name, title, negative, positive, caption=""):
        negative = [value for value in negative if _finite(value)]
        positive = [value for value in positive if _finite(value)]
        fig, ax = self._figure(title)
        bins = [index / 20 for index in range(21)]
        if negative:
            ax.hist(negative, bins=bins, color=COLORS[0], alpha=0.6, label=f"Explicit negative (n={len(negative):,})")
        if positive:
            ax.hist(positive, bins=bins, color=COLORS[1], alpha=0.6, label=f"Explicit positive (n={len(positive):,})")
        if negative or positive:
            self._legend(fig, ax)
        else:
            self._empty(ax)
        ax.set(xlabel="Predicted secret probability", ylabel="Labeled candidate count", xlim=(0, 1))
        ax.grid(axis="y", color="#E0E6EE", linewidth=0.8)
        self._save(fig, name, title, caption)
