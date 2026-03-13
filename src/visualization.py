from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())


def _load_json_list(file_path: str | Path) -> list[dict]:
    path = Path(file_path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, list) else []


def generate_visual_reports(
    confusion_rows: list[dict],
    quantitative_rows: list[dict],
    output_dir: str | Path,
    title_prefix: str,
) -> list[str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated_files: list[str] = []

    group_rows = [row for row in quantitative_rows if row.get("aggregate_type") == "group"]
    if group_rows:
        labels = [f"{row['prompt_strategy']}\n{row['model_id']}" for row in group_rows]
        x = np.arange(len(group_rows))
        width = 0.25

        precision = np.array([float(row.get("precision", 0.0)) for row in group_rows], dtype=float)
        accuracy = np.array([float(row.get("accuracy", 0.0)) for row in group_rows], dtype=float)
        f1 = np.array([float(row.get("f1", 0.0)) for row in group_rows], dtype=float)

        fig, ax = plt.subplots(figsize=(max(10, len(group_rows) * 1.15), 6))
        ax.bar(x - width, precision, width, label="Precision")
        ax.bar(x, accuracy, width, label="Accuracy")
        ax.bar(x + width, f1, width, label="F1")
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title(f"{title_prefix} - Precision/Accuracy/F1")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.legend(loc="lower right")
        fig.tight_layout()

        scores_file = out_dir / "scores_bar.png"
        fig.savefig(scores_file, dpi=150)
        plt.close(fig)
        generated_files.append(str(scores_file))

    for row in confusion_rows:
        model_id = str(row.get("model_id", "unknown_model"))
        strategy = str(row.get("prompt_strategy", "unknown_strategy"))

        tn = float(row.get("tn", 0))
        fp = float(row.get("fp", 0))
        fn = float(row.get("fn", 0))
        tp = float(row.get("tp", 0))

        matrix = np.array([[tn, fp], [fn, tp]], dtype=float)
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)

        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        heatmap = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks([0, 1], labels=["Judge Predicted Incorrect", "Judge Predicted Correct"])
        ax.set_yticks([0, 1], labels=["Actually Incorrect", "Actually Correct"])
        ax.set_title(f"{strategy} | {model_id}\nNormalized Confusion")

        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{normalized[i, j]:.2f}\n({int(matrix[i, j])})",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=9,
                )

        fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()

        confusion_file = out_dir / f"confusion_{_slug(strategy)}_{_slug(model_id)}.png"
        fig.savefig(confusion_file, dpi=150)
        plt.close(fig)
        generated_files.append(str(confusion_file))

    return generated_files


def generate_visual_reports_from_files(
    confusion_path: str | Path,
    quantitative_summary_path: str | Path,
    output_dir: str | Path,
    title_prefix: str,
) -> list[str]:
    confusion_rows = _load_json_list(confusion_path)
    quantitative_rows = _load_json_list(quantitative_summary_path)
    return generate_visual_reports(
        confusion_rows=confusion_rows,
        quantitative_rows=quantitative_rows,
        output_dir=output_dir,
        title_prefix=title_prefix,
    )