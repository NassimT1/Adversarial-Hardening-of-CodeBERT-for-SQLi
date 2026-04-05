from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_PATH = BASE_DIR / "data" / "judge_results.csv"
AUDIT_PATH = BASE_DIR / "data" / "manual_audit.csv"
SUMMARY_PATH = BASE_DIR / "data" / "judge_summary.txt"


def main():
    results = pd.read_csv(RESULTS_PATH)
    audit = pd.read_csv(AUDIT_PATH)
    merged = results.merge(audit, on="id", how="inner")

    merged["keep"] = merged["keep"].astype(str).str.lower().map({"true": True, "false": False})
    merged["manual_keep"] = merged["manual_keep"].astype(str).str.lower().map({"true": True, "false": False})

    agreement = (merged["keep"] == merged["manual_keep"]).mean() if len(merged) else 0.0
    accepted = int((results["keep"].astype(str).str.lower() == "true").sum())
    rejected = int((results["keep"].astype(str).str.lower() == "false").sum())

    lines = [
        f"Total judged: {len(results)}",
        f"Accepted by judge: {accepted}",
        f"Rejected by judge: {rejected}",
        f"Manual audit size: {len(merged)}",
        f"Agreement with manual labels: {agreement:.2%}",
        "",
        "Confusion table (judge vs manual):",
    ]

    if len(merged):
        ctab = pd.crosstab(merged["keep"], merged["manual_keep"], rownames=["judge_keep"], colnames=["manual_keep"])
        lines.extend(ctab.to_string().splitlines())
    else:
        lines.append("No overlapping IDs between judge_results.csv and manual_audit.csv")

    summary = "\n".join(lines)
    SUMMARY_PATH.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"\nSaved summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
