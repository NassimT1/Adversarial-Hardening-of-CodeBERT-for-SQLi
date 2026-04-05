"""
test_sandbox.py - evaluate sqli_sandbox against a feature extraction CSV

The script expects a CSV containing both the dataset (Query, Label) 
and the profile information from sqlglot:
    Query, Label, is_valid_syntax, winning_context_index, winning_dialect,
    tables, columns, literal_types, select_arm_widths, node_set, ast_sequence

Usage:
    python test_sandbox.py 
    python test_sandbox.py path/to/Feature_Extraction_Results.csv
    python test_sandbox.py feature_extraction_results.csv --limit 2000
    python test_sandbox.py feature_extraction_results.csv --show-fp --show-fn
"""

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sqli_sandbox import ASTProfile, SQLiSandbox


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate sqli_sandbox against an extracted features CSV"
    )
    p.add_argument(
        "csv_path",
        nargs="?",
        default="SQL_injection_Dataset_Feature_Extraction_Results.csv",
        help="Path to the Feature Extraction Results CSV",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N rows of the dataset",
    )
    p.add_argument(
        "--show-fp",
        action="store_true",
        help="Print false positive samples after the summary",
    )
    p.add_argument(
        "--show-fn",
        action="store_true",
        help="Print false negative samples after the summary",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=20,
        help="Maximum number of FP/FN samples to display (default: 20)",
    )
    return p.parse_args()


def load_profiles(profile_path):
    profiles = {}

    def safe_json(value, default):
        try:
            parsed = json.loads(value or "null")
            return parsed if parsed is not None else default
        except Exception:
            return default

    def safe_int(value, default):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    with open(profile_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = row.get("Query", "")
            if not query:
                continue

            is_valid = row.get("is_valid_syntax", "False").strip().lower() == "true"

            profiles[query] = ASTProfile(
                is_valid=is_valid,
                winning_context_index=safe_int(row.get("winning_context_index"), -1),
                winning_dialect=row.get("winning_dialect") or None,
                tables=safe_json(row.get("tables"), []),
                columns=safe_json(row.get("columns"), []),
                literal_types=safe_json(row.get("literal_types"), []),
                select_arm_widths=safe_json(row.get("select_arm_widths"), []),
                node_set=set(safe_json(row.get("node_set"), [])),
            )

    return profiles


def bar(value, width=30):
    filled = round(value * width)
    return "█" * filled + "░" * (width - filled)


def path_metrics(tp, tn, fp, fn):
    n = tp + tn + fp + fn
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / n if n else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return prec, rec, f1, acc, fpr


def main():
    args = parse_args()
    profile_path = Path(args.csv_path)

    if not profile_path.exists():
        print(f"ERROR: CSV not found at '{profile_path}'")
        sys.exit(1)

    try:
        profiles = load_profiles(profile_path)
    except Exception as exc:
        print(f"ERROR: could not load CSV - {exc}")
        sys.exit(1)

    # Load dataset directly from the feature extraction CSV
    with open(profile_path, newline="", encoding="utf-8-sig") as f:
        all_rows = list(csv.DictReader(f))

    rows = all_rows[: args.limit] if args.limit else all_rows
    total = len(rows)
    n_mal = sum(1 for r in rows if int(r["Label"]) == 1)
    n_ben = total - n_mal

    W = 60
    print(f"\n{'═' * W}")
    print(f"  Dataset : {profile_path.name}")
    print(f"  Rows    : {total:,}" + (f"  (of {len(all_rows):,})" if args.limit else ""))
    print(f"  Classes : {n_mal:,} malicious  |  {n_ben:,} benign")
    print(f"{'═' * W}\n")

    sb = SQLiSandbox()

    TP = TN = FP = FN = 0
    executed_count = static_count = 0
    executed_mal = executed_ben = 0
    static_mal = static_ben = 0
    ex_TP = ex_TN = ex_FP = ex_FN = 0
    st_TP = st_TN = st_FP = st_FN = 0
    unknown_count = 0

    exploit_counts = Counter()
    fp_rows = []
    fn_rows = []

    print("Running", end="", flush=True)
    t0 = time.time()
    tick = max(1, total // 20)

    for i, row in enumerate(rows):
        query = row["Query"]
        label = int(row["Label"])

        profile = profiles.get(query)
        if profile is None:
            continue

        result = sb.test(query, profile)
        pred = 1 if result.malicious else 0

        if label == 1 and pred == 1:
            TP += 1
            exploit_counts[result.exploit_type or "none"] += 1
            if result.exploit_type == "unknown_injection":
                unknown_count += 1
        elif label == 0 and pred == 0:
            TN += 1
        elif label == 0 and pred == 1:
            FP += 1
            fp_rows.append((query, result.exploit_type or "none", result.detection_reason))
        else:
            FN += 1
            fn_rows.append((query, result.exploit_type or "none", result.detection_reason))

        if result.executed:
            executed_count += 1
            executed_mal += int(result.malicious)
            executed_ben += int(not result.malicious)
            if label == 1 and pred == 1:
                ex_TP += 1
            elif label == 0 and pred == 0:
                ex_TN += 1
            elif label == 0 and pred == 1:
                ex_FP += 1
            else:
                ex_FN += 1
        else:
            static_count += 1
            static_mal += int(result.malicious)
            static_ben += int(not result.malicious)
            if label == 1 and pred == 1:
                st_TP += 1
            elif label == 0 and pred == 0:
                st_TN += 1
            elif label == 0 and pred == 1:
                st_FP += 1
            else:
                st_FN += 1

        if (i + 1) % tick == 0:
            print(".", end="", flush=True)

    elapsed = time.time() - t0
    print(f" done ({elapsed:.1f}s, {total / elapsed:.0f} q/s)\n")

    ex_prec, ex_rec, ex_f1, ex_acc, ex_fpr = path_metrics(ex_TP, ex_TN, ex_FP, ex_FN)
    st_prec, st_rec, st_f1, st_acc, st_fpr = path_metrics(st_TP, st_TN, st_FP, st_FN)

    print("─" * W)
    print(f"  SANDBOX EXECUTION PATH  ({executed_count:,} queries)")
    print("─" * W)
    print(f"  {'Accuracy':<22} {ex_acc:.4f}")
    print(f"  {'Precision':<22} {ex_prec:.4f}")
    print(f"  {'Recall':<22} {ex_rec:.4f}")
    print(f"  {'F1':<22} {ex_f1:.4f}")
    print(f"  {'False Positive Rate':<22} {ex_fpr:.4f}")
    print(f"  TP={ex_TP:,}  FP={ex_FP:,}  TN={ex_TN:,}  FN={ex_FN:,}")
    print(f"  Predicted malicious : {executed_mal:,}  |  Predicted benign : {executed_ben:,}")
    print("─" * W)
    print(f"  STATIC FILTER PATH  ({static_count:,} queries)")
    print("─" * W)
    print(f"  {'Accuracy':<22} {st_acc:.4f}")
    print(f"  {'Precision':<22} {st_prec:.4f}")
    print(f"  {'Recall':<22} {st_rec:.4f}")
    print(f"  {'F1':<22} {st_f1:.4f}")
    print(f"  {'False Positive Rate':<22} {st_fpr:.4f}")
    print(f"  TP={st_TP:,}  FP={st_FP:,}  TN={st_TN:,}  FN={st_FN:,}")
    print(f"  Predicted malicious : {static_mal:,}  |  Predicted benign : {static_ben:,}")
    print()

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (TP + TN) / total
    specificity = TN / (TN + FP) if (TN + FP) else 0.0
    fpr = FP / (FP + TN) if (FP + TN) else 0.0

    print("═" * W)
    print("  PERFORMANCE METRICS")
    print("─" * W)
    print(f"  {'Accuracy':<22} {accuracy:.4f}  {bar(accuracy)}")
    print(f"  {'Precision (PPV)':<22} {precision:.4f}  {bar(precision)}")
    print(f"  {'Recall (Sensitivity)':<22} {recall:.4f}  {bar(recall)}")
    print(f"  {'Specificity (TNR)':<22} {specificity:.4f}  {bar(specificity)}")
    print(f"  {'F1 Score':<22} {f1:.4f}  {bar(f1)}")
    print(f"  {'False Positive Rate':<22} {fpr:.4f}  {bar(fpr)}")
    print("─" * W)
    print("  CONFUSION MATRIX")
    print(f"                     Predicted")
    print(f"                  Malicious  Benign")
    print(f"  Actual Malicious  {TP:>7,}  {FN:>6,}   (detection rate: {recall * 100:.1f}%)")
    print(f"  Actual Benign     {FP:>7,}  {TN:>6,}   (false alarm rate: {fpr * 100:.1f}%)")
    print("═" * W)

    print("\n  DETECTION BREAKDOWN  (true positives only)")
    print("─" * W)
    for exploit, count in exploit_counts.most_common():
        pct = count / TP * 100 if TP else 0
        marker = "  ← unknown, needs review" if exploit == "unknown_injection" else ""
        print(f"  {exploit:<34} {count:>5,}  ({pct:.1f}%){marker}")
    print("═" * W)

    if unknown_count > 0:
        pct = unknown_count / TP * 100 if TP else 0
        print(f"\n  UNKNOWN INJECTION  ({unknown_count:,} true positives, {pct:.1f}% of TP)")
        print("─" * W)
        print(
            "  These queries were detected as suspicious (SQL content present)"
            "\n  but did not match any of our main injection categories."
            "\n  They should be reviewed to either add a new static pattern,"
            "\n  or confirm they are genuinely novel adversarial payloads."
        )
        print("═" * W)

    if args.show_fp and fp_rows:
        print(f"\n  FALSE POSITIVES ({FP:,}) - benign queries flagged as malicious")
        print("─" * W)
        fp_by_type = Counter(etype for _, etype, _ in fp_rows)
        print("  Breakdown by detected type (filter causing the FP):")
        for etype, cnt in fp_by_type.most_common():
            print(f"    {etype:<34} {cnt:>5,}")
        print()
        fp_by_reason = Counter(reason[:60] for _, _, reason in fp_rows)
        print("  Breakdown by detection reason:")
        for reason, cnt in fp_by_reason.most_common(8):
            print(f"    [{cnt:>4}]  {reason}")
        print(f"\n  Sample (first {min(args.sample, len(fp_rows))}):")
        for q, etype, reason in fp_rows[: args.sample]:
            print(f"    {repr(q[:70])}")
            print(f"         type   → {etype}")
            print(f"         reason → {reason[:65]}")

    if args.show_fn and fn_rows:
        print(f"\n  FALSE NEGATIVES ({FN:,}) - malicious queries missed by sandbox")
        print("─" * W)
        print(f"\n  Sample (first {min(args.sample, len(fn_rows))}):")
        for q, etype, reason in fn_rows[: args.sample]:
            print(f"    {repr(q[:70])}")
            print(f"         reason → {reason[:65]}")

    print(
        f"\n  Accuracy={accuracy * 100:.2f}%  F1={f1:.4f}  "
        f"Precision={precision:.4f}  Recall={recall:.4f}  FPR={fpr:.4f}"
    )
    print(f"  TP={TP:,}  FP={FP:,}  TN={TN:,}  FN={FN:,}\n")


if __name__ == "__main__":
    main()