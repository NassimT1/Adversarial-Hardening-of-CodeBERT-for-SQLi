from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize accepted and rejected judge outputs."
    )
    parser.add_argument(
        "--accepted",
        default=str(BASE_DIR / "data" / "accepted_candidates.csv"),
        help="Accepted CSV path.",
    )
    parser.add_argument(
        "--rejected",
        default=str(BASE_DIR / "data" / "rejected_candidates.csv"),
        help="Rejected CSV path.",
    )
    parser.add_argument(
        "--out",
        default=str(BASE_DIR / "data" / "t3_metrics_summary.md"),
        help="Markdown summary output path.",
    )
    return parser.parse_args()


def counts_to_lines(series: pd.Series) -> list[str]:
    if series.empty:
        return ["- none: 0"]
    return [f"- {idx}: {int(val)}" for idx, val in series.items()]


def top_text_counts(series: pd.Series, top_n: int = 10) -> pd.Series:
    cleaned = (
        series.fillna("")
        .astype(str)
        .str.strip()
        .replace("", "unknown")
    )
    return cleaned.value_counts().head(top_n)


def main():
    args = parse_args()

    accepted_path = Path(args.accepted)
    rejected_path = Path(args.rejected)
    output_path = Path(args.out)

    if not accepted_path.exists():
        raise FileNotFoundError(f"Accepted CSV not found: {accepted_path}")
    if not rejected_path.exists():
        raise FileNotFoundError(f"Rejected CSV not found: {rejected_path}")

    accepted_df = pd.read_csv(accepted_path)
    rejected_df = pd.read_csv(rejected_path)

    accepted_count = len(accepted_df)
    rejected_count = len(rejected_df)
    total_count = accepted_count + rejected_count
    acceptance_rate = (accepted_count / total_count * 100.0) if total_count else 0.0

    accepted_context = (
        accepted_df["template_context"].fillna("unknown").astype(str).value_counts()
        if "template_context" in accepted_df.columns
        else pd.Series(dtype="int64")
    )

    accepted_attack = (
        accepted_df["llm_attack_category"].fillna("unknown").astype(str).value_counts()
        if "llm_attack_category" in accepted_df.columns
        else pd.Series(dtype="int64")
    )

    accepted_quality = (
        accepted_df["judge_overall_quality_score"].fillna("unknown").astype(str).value_counts().sort_index()
        if "judge_overall_quality_score" in accepted_df.columns
        else pd.Series(dtype="int64")
    )

    accepted_reason_top = (
        top_text_counts(accepted_df["judge_reason"], top_n=8)
        if "judge_reason" in accepted_df.columns
        else pd.Series(dtype="int64")
    )

    rejected_stage = (
        rejected_df["failure_stage"].fillna("unknown").astype(str).value_counts()
        if "failure_stage" in rejected_df.columns
        else pd.Series(dtype="int64")
    )

    rejected_context = (
        rejected_df["template_context"].fillna("unknown").astype(str).value_counts()
        if "template_context" in rejected_df.columns
        else pd.Series(dtype="int64")
    )

    rejected_reason_top = (
        top_text_counts(rejected_df["failure_reason"], top_n=10)
        if "failure_reason" in rejected_df.columns
        else pd.Series(dtype="int64")
    )

    lines = []
    lines.append("# T3 Metrics Summary")
    lines.append("")
    lines.append("## Final dataset summary")
    lines.append(f"- Input accepted file: `{accepted_path.name}`")
    lines.append(f"- Input rejected file: `{rejected_path.name}`")
    lines.append(f"- Total accepted queries: **{accepted_count}**")
    lines.append(f"- Total rejected queries: **{rejected_count}**")
    lines.append(f"- Acceptance rate: **{acceptance_rate:.2f}%**")
    lines.append("")

    lines.append("## Accepted context breakdown")
    lines.extend(counts_to_lines(accepted_context))
    lines.append("")

    lines.append("## Accepted attack-category breakdown")
    lines.extend(counts_to_lines(accepted_attack))
    lines.append("")

    lines.append("## Accepted judge score distribution")
    lines.extend(counts_to_lines(accepted_quality))
    lines.append("")

    lines.append("## Most common acceptance reasons")
    lines.extend(counts_to_lines(accepted_reason_top))
    lines.append("")

    lines.append("## Rejected failure-stage breakdown")
    lines.extend(counts_to_lines(rejected_stage))
    lines.append("")

    lines.append("## Rejected context breakdown")
    lines.extend(counts_to_lines(rejected_context))
    lines.append("")

    lines.append("## Most common rejection reasons")
    lines.extend(counts_to_lines(rejected_reason_top))
    lines.append("")

    lines.append("## Interpretation")
    lines.append("- Accepted rows represent candidate SQLi samples that passed deterministic checks and the LLM judge rubric.")
    lines.append("- Rejected rows failed either deterministic validation (sandbox/AST) or the judge rubric for realism, maliciousness preservation, or non-triviality.")
    lines.append("- This file can be used directly in the report/presentation for the statistics and analysis section.")

    output_text = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_text, encoding="utf-8")
    print(output_text)


if __name__ == "__main__":
    main()