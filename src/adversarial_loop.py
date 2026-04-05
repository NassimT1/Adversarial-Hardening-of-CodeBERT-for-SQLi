"""
Main Adversarial Loop Controller
===========================================================
Connects every stage: AST validation, sandbox testing, CodeBERT detection, and judge
"""

import argparse
import json
import sys
from enum import Enum
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "LLM_JUDGE" / "src"))

from sqli_sandbox import ASTProfile, SQLiSandbox
from utils import heuristic_judge, apply_hard_rules
from judge_schema import JudgeResult


# ===========================================================================
# 1.  CONFIGURATION
# ===========================================================================

class LoopConfig:
    def __init__(self):
        self.payloads_csv = ROOT / "data" / "generated_payloads.csv"
        self.features_csv = ROOT / "data" / "generated_payloads_Feature_Extraction_Results.csv"
        self.output_csv = ROOT / "data" / "adversarial_dataset.csv"
        self.failures_json = ROOT / "data" / "mutation_queue.json"
        self.codebert_dir = ROOT / "models" / "codebert_sqli"

        self.max_mutations = 3
        self.codebert_threshold = 0.5
        self.judge_backend = "heuristic"
        self.max_tokens = 256
        self.cycle = 1
        self.max_cycles = 1


# ===========================================================================
# 2.  QUERY STATE MACHINE
# ===========================================================================

class Stage(str, Enum):
    PENDING = "pending"
    AST = "ast"
    SANDBOX = "sandbox"
    CODEBERT = "codebert"
    JUDGE = "judge"
    ACCEPTED = "accepted"
    DISCARDED = "discarded"


class QueryRecord:
    def __init__(self, query_id, payload, full_query, attack_category,
                 template_context, mutation_count=0, status=Stage.PENDING,
                 failure_stage=None, failure_reason="", codebert_score=0.0):
        self.query_id = query_id
        self.payload = payload
        self.full_query = full_query
        self.attack_category = attack_category
        self.template_context = template_context
        self.mutation_count = mutation_count
        self.status = status
        self.failure_stage = failure_stage
        self.failure_reason = failure_reason
        self.codebert_score = codebert_score

        self.ast_metadata = {}
        self.sandbox_metadata = {}
        self.judge_metadata = {}

    def fail(self, stage, reason, codebert_score=0.0):
        self.status = Stage.PENDING
        self.failure_stage = stage
        self.failure_reason = reason
        if codebert_score > 0:
            self.codebert_score = codebert_score

    def accept(self):
        self.status = Stage.ACCEPTED

    def discard(self, reason):
        self.status = Stage.DISCARDED
        self.failure_reason = reason

    # Convert to mutation request format for mutate_payloads.py
    def to_mutation_request(self):
        return {
            "query_id": self.query_id,
            "payload": self.payload,
            "full_query": self.full_query,
            "attack_category": self.attack_category,
            "template_context": self.template_context,
            "failure_stage": self.failure_stage.value if self.failure_stage else None,
            "failure_reason": self.failure_reason,
            "codebert_score": self.codebert_score,
            "mutation_count": self.mutation_count,
            "hint": _mutation_hint(self.failure_stage, self.codebert_score),
        }


def _mutation_hint(stage, codebert_score):
    hints = {
        Stage.AST:"The query has invalid SQL syntax. Fix the structure while keeping the attack",
        Stage.SANDBOX: "The sandbox returned benign. Try a different injection technique",
        Stage.CODEBERT: f"CodeBERT detected this with {codebert_score:.10%} confidence. Obfuscate to evade",
        Stage.JUDGE: "The judge rejected this. Make it more realistic with clear malicious intent",
    }
    return hints.get(stage, "Mutate the payload to improve its quality")


# ===========================================================================
# 3.  PIPELINE STAGES
# ===========================================================================

# Run AST feature extraction stage
def run_ast_stage(records, features_csv, max_mutations=3):
    def _load_profiles(csv_path):
        import csv as _csv
        profiles = {}

        def safe_json(v, default):
            try:
                p = json.loads(v or "null")
                return p if p is not None else default
            except Exception:
                return default

        def safe_int(v, default):
            try:
                return int(float(v))
            except (TypeError, ValueError):
                return default

        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            for row in _csv.DictReader(f):
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

    profiles = _load_profiles(features_csv)

    for rec in records:
        profile = profiles.get(rec.payload)
        if profile is None:
            rec.fail(Stage.AST, reason="Query not found in feature extraction CSV")
            print(f"    x [{rec.query_id}] attempt {rec.mutation_count + 1}/{max_mutations}"
                  f" | ast | not found in feature extraction CSV")
        else:
            rec.ast_metadata = {
                "ast_is_valid": profile.is_valid,
                "ast_dialect": profile.winning_dialect,
                "ast_node_set": json.dumps(list(profile.node_set)) if profile.node_set else "[]",
            }

    return profiles

# Run sandbox stage
def run_sandbox_stage(records, profiles, sandbox, max_mutations=3):
    pending = [r for r in records if r.status == Stage.PENDING and r.failure_stage is None]
    for rec in pending:
        profile = profiles.get(rec.payload)
        result = sandbox.test(rec.payload, profile)

        rec.sandbox_metadata = {
            "sandbox_executed": getattr(result, "executed", False),
            "sandbox_detection_mode": getattr(result, "mode", "none"),
            "sandbox_exploit_type": getattr(result, "exploit_type", "none"),
        }

        if not result.malicious:
            rec.fail(Stage.SANDBOX,
                     reason=f"Sandbox returned benign. Reason: {result.detection_reason}")
            print(f"    x [{rec.query_id}] attempt {rec.mutation_count + 1}/{max_mutations}"
                  f" | sandbox | {result.detection_reason}")


# Define CodeBERT-based detection stage
class CodeBERTDetector:
    def __init__(self, model_dir, max_tokens=256):
        print(f"  Loading CodeBERT from {model_dir} …")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.eval()
        self.max_tokens = max_tokens
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"  CodeBERT ready on {self.device}")

    def score(self, texts):
        inputs = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_tokens, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        return probs[:, 1].cpu().tolist()


# Run CodeBERT detection stage
def run_codebert_stage(records, detector, threshold, batch_size=32, max_mutations=3):
    pending = [r for r in records if r.status == Stage.PENDING and r.failure_stage is None]
    if not pending:
        return

    for i in range(0, len(pending), batch_size):
        batch  = pending[i : i + batch_size]
        scores = detector.score([r.full_query for r in batch])
        for rec, score in zip(batch, scores):
            rec.codebert_score = score
            if score > threshold:
                rec.fail(Stage.CODEBERT,
                         reason=f"CodeBERT detected injection with {score:.2%} confidence",
                         codebert_score=score)
                print(f"    x [{rec.query_id}] attempt {rec.mutation_count + 1}/{max_mutations}"
                      f" | codebert | detected with {score:.2%} confidence")


# Run judge stage
def run_judge_stage(records, backend="heuristic", max_mutations=3):
    pending = [r for r in records if r.status == Stage.PENDING and r.failure_stage is None]
    if not pending:
        return

    for rec in pending:
        row = {
            "query_id": rec.query_id,
            "payload": rec.payload,
            "full_query": rec.full_query,
            "template_context": rec.template_context,
            "llm_attack_category": rec.attack_category,
            "ast_is_valid": bool(rec.ast_metadata.get("ast_is_valid", True)),
            "ast_dialect": str(rec.ast_metadata.get("ast_dialect", "")),
            "ast_node_set": str(rec.ast_metadata.get("ast_node_set", "[]")),
            "sandbox_executed": bool(rec.sandbox_metadata.get("sandbox_executed", True)),
            "sandbox_detection_mode": str(rec.sandbox_metadata.get("sandbox_detection_mode", "")),
            "sandbox_exploit_type": str(rec.sandbox_metadata.get("sandbox_exploit_type", "")),
            "seed_payload": "",
            "seed_payload_available": False,
            "notes": f"attack_category={rec.attack_category}",
        }
        try:
            if backend == "heuristic":
                parsed, _ = heuristic_judge(row)
                parsed = apply_hard_rules(row, parsed)
            else:
                from utils import (ollama_local_generate, build_user_prompt, extract_json, load_system_prompt)
                raw = ollama_local_generate(load_system_prompt(), build_user_prompt(row))
                parsed = apply_hard_rules(row, extract_json(raw))

            result = JudgeResult(**parsed)
            rec.judge_metadata = {
                "judge_malicious_intent_preserved": result.malicious_intent_preserved,
                "judge_realistic_for_context": result.realistic_for_context,
                "judge_non_trivial_mutation": result.non_trivial_mutation,
                "judge_overall_quality_score": result.overall_quality_score,
                "judge_reason": result.reason,
            }

            if not result.keep:
                rec.fail(Stage.JUDGE, reason=f"Judge rejected: {result.reason}")
                print(f"    x [{rec.query_id}] attempt {rec.mutation_count + 1}/{max_mutations}"
                      f" | judge | {result.reason[:80]}")
            else:
                rec.accept()
                print(f"    ✓ [{rec.query_id}] accepted"
                      f" | mutation {rec.mutation_count}"
                      f" | codebert={rec.codebert_score:.10f}")

        except Exception as exc:
            rec.fail(Stage.JUDGE, reason=f"Judge error: {exc}")
            print(f"    x [{rec.query_id}] attempt {rec.mutation_count + 1}/{max_mutations}"
                  f" | judge | error: {exc.__class__.__name__}: {str(exc)[:60]}")


# ===========================================================================
# 4.  FILE I/O
# ===========================================================================

# Get current count of accepted samples in adversarial_dataset.csv
def _get_next_id(output_csv):
    if not output_csv.exists() or output_csv.stat().st_size == 0:
        return 0
    try:
        df = pd.read_csv(output_csv, usecols=["query_id"])
        # Extract the numeric part from q00000 format
        numeric = (
            df["query_id"]
            .astype(str)
            .str.extract(r"^q(\d+)$")[0]
            .dropna()
            .astype(int)
        )
        return int(numeric.max()) + 1 if not numeric.empty else 0
    except Exception:
        return 0


# Load initial payloads from generated_payloads.csv and assign unique query_ids
def load_initial_records(payloads_csv, output_csv):
    df = pd.read_csv(payloads_csv)
    start_id = _get_next_id(output_csv)
    records = []

    for i, row in df.iterrows():
        query_id = f"q{start_id + len(records):05d}"

        mutation_count = (
            int(row["mutation_count"])
            if "mutation_count" in row and pd.notna(row["mutation_count"])
            else 0
        )

        records.append(QueryRecord(
            query_id=query_id,
            payload=str(row["payload"]),
            full_query=str(row["full_query"]),
            attack_category=str(row["attack_category"]),
            template_context=str(row["template_context"]),
            mutation_count=mutation_count,
        ))

    return records


# Save accepted records to adversarial_dataset.csv (append mode)
def save_results(records, output_csv):
    accepted = [r for r in records if r.status == Stage.ACCEPTED]
    if not accepted:
        print("  No accepted records to save")
        return

    rows = []
    for r in accepted:
        row_data = {
            "query_id": r.query_id,
            "payload": r.payload,
            "full_query": r.full_query,
            "llm_attack_category": r.attack_category,
            "template_context": r.template_context,
            "label": 1,
            "mutation_count": r.mutation_count,
            "codebert_score": round(r.codebert_score, 4),
        }
        row_data.update(r.ast_metadata)
        row_data.update(r.sandbox_metadata)
        row_data.update(r.judge_metadata)
        rows.append(row_data)

    file_path = Path(output_csv)
    write_header = not file_path.exists() or file_path.stat().st_size == 0

    pd.DataFrame(rows).to_csv(file_path, mode="a", index=False, header=write_header)
    print(f"  Appended {len(rows)} accepted records -> {file_path}")


# Save mutation queue to JSON for mutate_payloads.py
def save_mutation_queue(queue, failures_json):
    with open(failures_json, "w", encoding="utf-8") as f:
        json.dump(queue, f, indent=2, ensure_ascii=False)
    print(f"  Mutation queue ({len(queue)} items) -> {failures_json}")


# Print status summary for current iteration
def print_status(records, iteration):
    counts = {s: 0 for s in Stage}
    for r in records:
        counts[r.status] += 1
    print(
        f"\n  [Iteration {iteration}] "
        f"accepted={counts[Stage.ACCEPTED]}  "
        f"pending={counts[Stage.PENDING]}  "
        f"discarded={counts[Stage.DISCARDED]}"
    )


# ===========================================================================
# 5.  MAIN LOOP
# ===========================================================================

def run_adversarial_loop(config):
    print("=" * 60)
    print("  STARTING ADVERSARIAL LOOP")
    print("=" * 60)

    sandbox  = SQLiSandbox()
    detector = CodeBERTDetector(config.codebert_dir, max_tokens=config.max_tokens)

    print(f"\n  Loading payloads from {config.payloads_csv}")
    # Pass output_csv so load_initial_records can determine the next ID
    all_records = load_initial_records(config.payloads_csv, config.output_csv)
    print(f"  Loaded {len(all_records)} queries")

    pending = [r for r in all_records if r.status == Stage.PENDING]
    if not pending:
        print("  No pending records to evaluate")
    else:
        print(f"\n{chr(9472)*60}\n  CYCLE {config.cycle}/{config.max_cycles}  ({len(pending)} pending)\n{chr(9472)*60}")

        print("  -> AST stage …")
        profiles = run_ast_stage(pending, config.features_csv,
                                 max_mutations=config.max_mutations)

        print("  -> Sandbox stage …")
        run_sandbox_stage(pending, profiles, sandbox,
                          max_mutations=config.max_mutations)

        print("  -> CodeBERT stage …")
        run_codebert_stage(pending, detector, threshold=config.codebert_threshold,
                           max_mutations=config.max_mutations)

        print("  -> Judge stage …")
        run_judge_stage(pending, backend=config.judge_backend,
                        max_mutations=config.max_mutations)

        # Get rid of queries that have failed/mutated too many times (mutation_count >= max_mutations) 
        for rec in all_records:
            if rec.status == Stage.PENDING and rec.mutation_count >= config.max_mutations:
                rec.discard(
                    reason=f"Discarded after {rec.mutation_count} mutations. "
                           f"Last failure: {rec.failure_stage}"
                )
                print(f"    ⊘ [{rec.query_id}] discarded"
                      f" | reached limit {rec.mutation_count}/{config.max_mutations} mutation attempts"
                      f" | last failure: {rec.failure_stage.value if rec.failure_stage else 'unknown'}")

        print_status(all_records, iteration=config.cycle)

        queue = [r.to_mutation_request() for r in all_records if r.status == Stage.PENDING]
        if queue:
            save_mutation_queue(queue, config.failures_json)
            print(f"\n  {len(queue)} queries queued for mutation")

    accepted  = [r for r in all_records if r.status == Stage.ACCEPTED]
    discarded = [r for r in all_records if r.status == Stage.DISCARDED]
    print(f"\n{'='*60}\n  RESULTS\n{'─'*60}")
    print(f"  Total    : {len(all_records)}")
    print(f"  Accepted : {len(accepted)}")
    print(f"  Discarded: {len(discarded)}")
    print(f"{'='*60}\n")

    save_results(all_records, config.output_csv)


# ===========================================================================
# 6.  CLI
# ===========================================================================

def parse_args():
    cfg = LoopConfig()
    p   = argparse.ArgumentParser(description="Adversarial Loop Orchestrator")
    p.add_argument("--payloads", default=str(cfg.payloads_csv))
    p.add_argument("--features", default=str(cfg.features_csv))
    p.add_argument("--output", default=str(cfg.output_csv))
    p.add_argument("--failures", default=str(cfg.failures_json))
    p.add_argument("--model", default=str(cfg.codebert_dir))
    p.add_argument("--confidence-threshold", type=float, default=cfg.codebert_threshold)
    p.add_argument("--max-mutations", type=int,   default=cfg.max_mutations)
    p.add_argument("--judge-backend", choices=["heuristic", "ollama_local"],
                                             default=cfg.judge_backend)
    p.add_argument("--cycle", type=int, default=cfg.cycle,
                   help="Current pipeline cycle number (passed by run_pipeline.py)")
    p.add_argument("--max-cycles", type=int, default=cfg.max_cycles,
                   help="Total pipeline cycle limit (passed by run_pipeline.py)")
    args = p.parse_args()

    cfg.payloads_csv = Path(args.payloads)
    cfg.features_csv = Path(args.features)
    cfg.output_csv = Path(args.output)
    cfg.failures_json = Path(args.failures)
    cfg.codebert_dir = Path(args.model)
    cfg.codebert_threshold = args.confidence_threshold
    cfg.max_mutations = args.max_mutations
    cfg.judge_backend = args.judge_backend
    cfg.cycle = args.cycle
    cfg.max_cycles = args.max_cycles
    return cfg


if __name__ == "__main__":
    run_adversarial_loop(parse_args())