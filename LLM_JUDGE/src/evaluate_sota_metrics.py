from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "evaluation_outputs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute SOTA-style metrics from evaluated prediction CSVs."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Evaluation CSV path. Repeat for multiple splits.",
    )
    parser.add_argument(
        "--name",
        action="append",
        required=True,
        help="Display name for each evaluation split. Repeat in the same order as --input.",
    )
    parser.add_argument(
        "--label-col",
        default="label",
        help="Ground-truth label column. Default: label",
    )
    parser.add_argument(
        "--pred-col",
        default="pred_label",
        help="Predicted label column. Default: pred_label",
    )
    parser.add_argument(
        "--score-col",
        default="pred_score",
        help="Positive-class score/probability column. Default: pred_score",
    )
    parser.add_argument(
        "--positive-label",
        default="1",
        help="Positive/malicious label value. Default: 1",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def normalize_scalar(value: Any) -> str:
    return str(value).strip().lower()


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def make_binary(series: pd.Series, positive_label: str) -> pd.Series:
    positive_norm = normalize_scalar(positive_label)
    return series.astype(str).map(lambda x: 1 if normalize_scalar(x) == positive_norm else 0)


def detect_score_column(df: pd.DataFrame, requested: str) -> str | None:
    if requested in df.columns:
        return requested
    for col in ["pred_score", "score", "probability", "prob", "confidence", "codebert_score"]:
        if col in df.columns:
            return col
    return None


def save_confusion_matrix_image(out_path: Path, split_name: str, cm: dict[str, int], normalize: bool = False):
    matrix = [
        [cm["tn"], cm["fp"]],
        [cm["fn"], cm["tp"]],
    ]

    if normalize:
        row_sums = []
        for row in matrix:
            s = sum(row)
            row_sums.append(s if s != 0 else 1)
        matrix = [
            [matrix[0][0] / row_sums[0], matrix[0][1] / row_sums[0]],
            [matrix[1][0] / row_sums[1], matrix[1][1] / row_sums[1]],
        ]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix)
    fig.colorbar(im, ax=ax)

    ax.set_title(split_name if not normalize else f"{split_name} (normalized)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["benign (0)", "malicious (1)"])
    ax.set_yticklabels(["benign (0)", "malicious (1)"])

    for i in range(2):
        for j in range(2):
            text_value = f"{matrix[i][j]:.2f}" if normalize else f"{int(matrix[i][j])}"
            ax.text(j, i, text_value, ha="center", va="center")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_placeholder_image(out_path: Path, split_name: str, reason: str):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        f"{split_name}\n\n{reason}",
        ha="center",
        va="center",
        wrap=True,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve_image(y_true, y_score, out_path: Path, split_name: str):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {split_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return roc_auc


def evaluate_one_split(
    csv_path: Path,
    split_name: str,
    label_col: str,
    pred_col: str,
    score_col: str,
    positive_label: str,
    output_dir: Path,
) -> dict[str, Any]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if label_col not in df.columns:
        raise ValueError(f"Missing label column '{label_col}' in {csv_path}")
    if pred_col not in df.columns:
        raise ValueError(f"Missing prediction column '{pred_col}' in {csv_path}")

    y_true = make_binary(df[label_col], positive_label)
    y_pred = make_binary(df[pred_col], positive_label)

    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    cm_array = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm = {
        "tn": int(cm_array[0, 0]),
        "fp": int(cm_array[0, 1]),
        "fn": int(cm_array[1, 0]),
        "tp": int(cm_array[1, 1]),
    }

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    actual_score_col = detect_score_column(df, score_col)
    auc_roc = None
    roc_curve_path = None

    if actual_score_col and y_true.nunique() == 2:
        y_score = pd.to_numeric(df[actual_score_col], errors="coerce").fillna(0.0)
        roc_curve_path = split_dir / "roc_curve.png"
        auc_roc = save_roc_curve_image(y_true, y_score, roc_curve_path, split_name)
    else:
        save_placeholder_image(
            split_dir / "roc_curve_unavailable.png",
            split_name,
            "ROC curve unavailable because a usable score column was not found or only one class is present.",
        )

    raw_cm_path = split_dir / "confusion_matrix.png"
    norm_cm_path = split_dir / "confusion_matrix_normalized.png"

    save_confusion_matrix_image(raw_cm_path, split_name, cm, normalize=False)
    save_confusion_matrix_image(norm_cm_path, split_name, cm, normalize=True)

    malicious_total = cm["tp"] + cm["fn"]

    metrics = {
        "split_name": split_name,
        "input_csv": str(csv_path),
        "num_rows": int(len(df)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "auc_roc": None if auc_roc is None else float(auc_roc),
        "attack_success_rate": float(safe_div(cm["fn"], malicious_total)),
        "detection_rate": float(safe_div(cm["tp"], malicious_total)),
        "label_column": label_col,
        "prediction_column": pred_col,
        "score_column": actual_score_col,
        "positive_label": positive_label,
        "confusion_matrix": cm,
        "artifacts": {
            "split_dir": str(split_dir),
            "confusion_matrix_png": str(raw_cm_path),
            "confusion_matrix_normalized_png": str(norm_cm_path),
            "roc_curve_png": None if roc_curve_path is None else str(roc_curve_path),
        },
    }

    with (split_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    pd.DataFrame(
        [
            {
                "split_name": split_name,
                "num_rows": metrics["num_rows"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "auc_roc": metrics["auc_roc"],
                "attack_success_rate": metrics["attack_success_rate"],
                "detection_rate": metrics["detection_rate"],
                "tn": cm["tn"],
                "fp": cm["fp"],
                "fn": cm["fn"],
                "tp": cm["tp"],
            }
        ]
    ).to_csv(split_dir / "metrics.csv", index=False)

    return metrics


def main():
    args = parse_args()

    if len(args.input) != len(args.name):
        raise ValueError("The number of --input arguments must match the number of --name arguments.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for input_path, split_name in zip(args.input, args.name):
        metrics = evaluate_one_split(
            csv_path=Path(input_path),
            split_name=split_name,
            label_col=args.label_col,
            pred_col=args.pred_col,
            score_col=args.score_col,
            positive_label=args.positive_label,
            output_dir=output_dir,
        )
        all_metrics.append(metrics)

    summary_rows = []
    for item in all_metrics:
        summary_rows.append(
            {
                "split_name": item["split_name"],
                "input_csv": item["input_csv"],
                "num_rows": item["num_rows"],
                "accuracy": item["accuracy"],
                "precision": item["precision"],
                "recall": item["recall"],
                "f1_score": item["f1_score"],
                "auc_roc": item["auc_roc"],
                "attack_success_rate": item["attack_success_rate"],
                "detection_rate": item["detection_rate"],
                "tn": item["confusion_matrix"]["tn"],
                "fp": item["confusion_matrix"]["fp"],
                "fn": item["confusion_matrix"]["fn"],
                "tp": item["confusion_matrix"]["tp"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / "all_metrics_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    with (output_dir / "all_metrics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"splits": summary_rows}, handle, indent=2, ensure_ascii=False)

    print("\nSaved evaluation outputs:")
    for item in all_metrics:
        print(f"- {item['split_name']}: {item['artifacts']['split_dir']}")
        print(
            f"  accuracy={item['accuracy']:.4f}, "
            f"precision={item['precision']:.4f}, "
            f"recall={item['recall']:.4f}, "
            f"f1={item['f1_score']:.4f}, "
            f"auc_roc={item['auc_roc'] if item['auc_roc'] is not None else 'N/A'}"
        )

    print(f"\nCombined summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()