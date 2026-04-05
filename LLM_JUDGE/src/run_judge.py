from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from judge_schema import JudgeResult
from utils import (
    apply_hard_rules,
    build_user_prompt,
    extract_json,
    heuristic_judge,
    load_system_prompt,
    normalize_candidate_row,
    ollama_local_generate,
    rejection_hint,
    sandbox_failure_reason,
)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

FINAL_ACCEPTED_COLUMNS = [
    "query_id",
    "payload",
    "full_query",
    "llm_attack_category",
    "template_context",
    "label",
    "mutation_count",
    "codebert_score",
    "ast_is_valid",
    "ast_dialect",
    "ast_node_set",
    "sandbox_executed",
    "sandbox_detection_mode",
    "sandbox_exploit_type",
    "judge_malicious_intent_preserved",
    "judge_realistic_for_context",
    "judge_non_trivial_mutation",
    "judge_overall_quality_score",
    "judge_reason",
]

FINAL_REJECTED_COLUMNS = [
    "query_id",
    "seed_payload",
    "payload",
    "full_query",
    "attack_category",
    "template_context",
    "failure_stage",
    "failure_reason",
    "codebert_score",
    "mutation_count",
    "hint",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LLM-as-a-Judge on one teammate-generated CSV and split it into accepted and rejected CSV files."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Teammate-generated candidate CSV input.",
    )
    parser.add_argument(
        "--backend",
        choices=["heuristic", "ollama_local"],
        default="ollama_local",
        help="Judge backend. Default: ollama_local",
    )
    parser.add_argument(
        "--accepted",
        default=str(BASE_DIR / "data" / "accepted_candidates.csv"),
        help="Accepted output CSV path.",
    )
    parser.add_argument(
        "--rejected",
        default=str(BASE_DIR / "data" / "rejected_candidates.csv"),
        help="Rejected output CSV path.",
    )
    parser.add_argument(
        "--all-results",
        default=str(BASE_DIR / "data" / "judge_results.csv"),
        help="All judge results CSV path.",
    )
    parser.add_argument(
        "--summary",
        default=str(BASE_DIR / "data" / "judge_summary.txt"),
        help="Basic run summary text file path.",
    )
    return parser.parse_args()


def _accepted_row(row: dict, result: JudgeResult) -> dict:
    return {
        "query_id": row["query_id"],
        "payload": row["payload"],
        "full_query": row["full_query"],
        "llm_attack_category": row["llm_attack_category"],
        "template_context": row["template_context"],
        "label": row["label"],
        "mutation_count": row["mutation_count"],
        "codebert_score": row["codebert_score"],
        "ast_is_valid": row["ast_is_valid"],
        "ast_dialect": row["ast_dialect"],
        "ast_node_set": row["ast_node_set"],
        "sandbox_executed": row["sandbox_executed"],
        "sandbox_detection_mode": row["sandbox_detection_mode"],
        "sandbox_exploit_type": row["sandbox_exploit_type"],
        "judge_malicious_intent_preserved": result.malicious_intent_preserved,
        "judge_realistic_for_context": result.realistic_for_context,
        "judge_non_trivial_mutation": result.non_trivial_mutation,
        "judge_overall_quality_score": result.overall_quality_score,
        "judge_reason": result.reason,
    }


def _reject_row(row: dict, stage: str, reason: str) -> dict:
    return {
        "query_id": row["query_id"],
        "seed_payload": row["seed_payload"] if row.get("seed_payload_available") else "",
        "payload": row["payload"],
        "full_query": row["full_query"],
        "attack_category": row["llm_attack_category"],
        "template_context": row["template_context"],
        "failure_stage": stage,
        "failure_reason": reason,
        "codebert_score": row["codebert_score"],
        "mutation_count": row["mutation_count"],
        "hint": rejection_hint(stage),
    }


def _all_result_row(row: dict, parsed: dict, backend_used: str, status: str) -> dict:
    return {
        "query_id": row["query_id"],
        "seed_payload": row["seed_payload"] if row.get("seed_payload_available") else "",
        "payload": row["payload"],
        "full_query": row["full_query"],
        "llm_attack_category": row["llm_attack_category"],
        "template_context": row["template_context"],
        "label": row["label"],
        "mutation_count": row["mutation_count"],
        "codebert_score": row["codebert_score"],
        "ast_is_valid": row["ast_is_valid"],
        "ast_dialect": row["ast_dialect"],
        "ast_node_set": row["ast_node_set"],
        "sandbox_executed": row["sandbox_executed"],
        "sandbox_detection_mode": row["sandbox_detection_mode"],
        "sandbox_exploit_type": row["sandbox_exploit_type"],
        "judge_malicious_intent_preserved": parsed["malicious_intent_preserved"],
        "judge_realistic_for_context": parsed["realistic_for_context"],
        "judge_non_trivial_mutation": parsed["non_trivial_mutation"],
        "judge_overall_quality_score": parsed["overall_quality_score"],
        "judge_keep": parsed["keep"],
        "judge_reason": parsed["reason"],
        "backend": backend_used,
        "status": status,
    }


def main():
    args = parse_args()

    input_path = Path(args.input)
    accepted_path = Path(args.accepted)
    rejected_path = Path(args.rejected)
    all_results_path = Path(args.all_results)
    summary_path = Path(args.summary)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.resolve() == accepted_path.resolve():
        raise ValueError("Accepted output path cannot be the same as the input path.")
    if input_path.resolve() == rejected_path.resolve():
        raise ValueError("Rejected output path cannot be the same as the input path.")
    if input_path.resolve() == all_results_path.resolve():
        raise ValueError("All-results output path cannot be the same as the input path.")

    system_prompt = load_system_prompt()

    df = pd.read_csv(input_path, dtype=str).fillna("")
    rows = [normalize_candidate_row(record) for record in df.to_dict(orient="records")]

    accepted_rows: list[dict] = []
    rejected_rows: list[dict] = []
    all_rows: list[dict] = []
    errors: list[dict] = []

    for row in tqdm(rows, desc="Judging candidates"):
        try:
            if row["existing_failure_stage"] and row["existing_failure_reason"]:
                rejected_rows.append(
                    _reject_row(row, row["existing_failure_stage"], row["existing_failure_reason"])
                )
                continue

            if not row["ast_is_valid"] or not row["sandbox_executed"]:
                stage = "sandbox"
                reason = sandbox_failure_reason(row)
                rejected_rows.append(_reject_row(row, stage, reason))

                heuristic_result, heuristic_backend = heuristic_judge(row)
                heuristic_result = apply_hard_rules(row, heuristic_result)
                all_rows.append(_all_result_row(row, heuristic_result, heuristic_backend, "rejected_pre_judge"))
                continue

            if args.backend == "heuristic":
                parsed, backend_used = heuristic_judge(row)
                parsed = apply_hard_rules(row, parsed)
            else:
                user_prompt = build_user_prompt(row)
                try:
                    raw_response = ollama_local_generate(system_prompt, user_prompt)
                    parsed = extract_json(raw_response)
                    parsed = apply_hard_rules(row, parsed)
                    backend_used = "ollama_local+rules"
                except Exception as exc:
                    parsed, _ = heuristic_judge(row)
                    parsed = apply_hard_rules(row, parsed)
                    backend_used = f"heuristic_fallback+rules ({exc.__class__.__name__})"

            result = JudgeResult(**parsed)
            all_rows.append(_all_result_row(row, parsed, backend_used, "ok"))

            if result.keep:
                accepted_rows.append(_accepted_row(row, result))
            else:
                rejected_rows.append(_reject_row(row, "judge", f"Judge rejected: {result.reason}"))

        except Exception as exc:
            errors.append({"query_id": row.get("query_id", ""), "error": str(exc)})
            rejected_rows.append(
                _reject_row(row, "judge", f"Judge failed to process the candidate cleanly. Error: {exc}")
            )

    accepted_df = pd.DataFrame(accepted_rows, columns=FINAL_ACCEPTED_COLUMNS)
    rejected_df = pd.DataFrame(rejected_rows, columns=FINAL_REJECTED_COLUMNS)
    all_results_df = pd.DataFrame(all_rows)

    for path in [accepted_path, rejected_path, all_results_path, summary_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    accepted_df.to_csv(accepted_path, index=False)
    rejected_df.to_csv(rejected_path, index=False)

    if not all_results_df.empty:
        all_results_df.to_csv(all_results_path, index=False)

    judge_reject_count = sum(1 for row in rejected_rows if row["failure_stage"] == "judge")
    sandbox_reject_count = sum(1 for row in rejected_rows if row["failure_stage"] == "sandbox")
    total = len(accepted_rows) + len(rejected_rows)
    acceptance_rate = (len(accepted_rows) / total * 100.0) if total else 0.0

    summary_lines = [
        f"Input rows: {len(rows)}",
        f"Accepted rows: {len(accepted_rows)}",
        f"Rejected rows: {len(rejected_rows)}",
        f"Acceptance rate: {acceptance_rate:.2f}%",
        f"Judge-stage rejects: {judge_reject_count}",
        f"Sandbox-stage rejects: {sandbox_reject_count}",
        f"Accepted CSV: {accepted_path}",
        f"Rejected CSV: {rejected_path}",
        f"All-results CSV: {all_results_path}",
        f"Errors captured: {len(errors)}",
    ]
    summary_text = "\n".join(summary_lines)
    summary_path.write_text(summary_text, encoding="utf-8")
    print(summary_text)

    if errors:
        print("\nSome rows produced exceptions and were pushed to the rejected output.")


if __name__ == "__main__":
    main()