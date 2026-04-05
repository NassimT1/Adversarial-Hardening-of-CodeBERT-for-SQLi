"""
Pipeline Runner
=====================================
Loop that controls the adversarial payload generation and validation process until either the
adversarial_dataset.csv reaches TARGET or MAX_CYCLES

Each cycle:
1. 
 If a mutation queue exists:
    mutate_payloads.py
 Else:
    phase1_llm_generate_payloads.py
2. Deduplicate generated_payloads.csv against adversarial_dataset.csv
3. feature_extraction.py -> sqlglot AST profiling
4. adversarial_loop.py -> sandbox + CodeBERT + judge
"""

import gc
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import pandas as pd

TARGET = 1000 # stop when this many rows are in adversarial_dataset.csv
MAX_CYCLES = 50 # stop to prevent infinite loop
DATA_DIR = Path("data")
PAYLOADS_CSV = DATA_DIR / "generated_payloads.csv"
FEATURES_CSV = DATA_DIR / "generated_payloads_Feature_Extraction_Results.csv"
DATASET_CSV = DATA_DIR / "adversarial_dataset.csv"
QUEUE_JSON = DATA_DIR / "mutation_queue.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_accepted_count():
    if not DATASET_CSV.exists():
        return 0
    try:
        return len(pd.read_csv(DATASET_CSV))
    except pd.errors.EmptyDataError:
        return 0


def queue_is_non_empty():
    return QUEUE_JSON.exists() and QUEUE_JSON.stat().st_size > 10


def run(cmd, label):
    print(f"\n  [{label}] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

# Remove duplicates from generated_payloads.csv that are already in adversarial_dataset.csv
def deduplicate_payloads():
    if not PAYLOADS_CSV.exists():
        return 0

    df_new = pd.read_csv(PAYLOADS_CSV)

    if DATASET_CSV.exists() and DATASET_CSV.stat().st_size > 0:
        df_existing = pd.read_csv(DATASET_CSV, usecols=["payload"])
        seen = set(df_existing["payload"].astype(str).str.strip())
        del df_existing   # release memory / file reference

        before = len(df_new)
        df_new = df_new[~df_new["payload"].astype(str).str.strip().isin(seen)]
        removed = before - len(df_new)
        if removed:
            print(f"  Deduplication: removed {removed} payload(s) already in the dataset.")

    if df_new.empty:
        print("  All generated payloads were duplicates — skipping this cycle's validation.")
        df_new.to_csv(PAYLOADS_CSV, index=False)
        del df_new
        gc.collect()
        return 0

    # Write deduplicated DataFrame to a temp file, then atomically replace the original
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=PAYLOADS_CSV.parent, suffix=".tmp"
    )
    try:
        os.close(tmp_fd)                          # close the OS-level fd
        df_new.to_csv(tmp_path, index=False)      # write to temp
        del df_new                                # drop DataFrame reference
        gc.collect()                              # force GC before rename
        os.replace(tmp_path, PAYLOADS_CSV)        # atomic replace on Windows
    except Exception:
        os.unlink(tmp_path)
        raise

    # Small delay to ensure file system updates are visible to subprocesses
    time.sleep(0.5)

    # Re-read just to count rows (file is now closed and unlocked)
    try:
        return len(pd.read_csv(PAYLOADS_CSV))
    except pd.errors.EmptyDataError:
        return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*60}")
    print(f"  PIPELINE RUNNER")
    print(f"  Target    : {TARGET} accepted samples")
    print(f"  Max cycles: {MAX_CYCLES}")
    print(f"{'='*60}")

    for cycle in range(1, MAX_CYCLES + 1):
        current = get_accepted_count()

        print(f"\n{'─'*60}")
        print(f"  Cycle {cycle}/{MAX_CYCLES}  |  Progress: {current}/{TARGET} accepted")
        print(f"{'─'*60}")

        if current >= TARGET:
            print(f"\n  Target reached ({current} >= {TARGET}). Done.")
            break

        # Step 1: Generate or mutate 
        if queue_is_non_empty():
            print("  Queue found → running mutation script…")
            run([sys.executable, "src/mutate_payloads.py"], "MUTATE")
        else:
            print("  Queue empty → generating new payload batch…")
            run([sys.executable, "phase1_llm_generate_payloads.py"], "GENERATE")

        # Step 2: Deduplicate against existing dataset
        remaining = deduplicate_payloads()
        if remaining == 0:
            print("  No unique payloads to validate. Skipping to next cycle.")
            time.sleep(2)
            continue

        # Step 3: Extract AST features 
        run([
            sys.executable, "src/feature_extraction.py",
            "--input",  str(PAYLOADS_CSV),
            "--output", str(FEATURES_CSV),
        ], "FEATURES")

        # Step 4: Run pipeline (sandbox -> CodeBERT -> judge) 
        run([
            sys.executable, "src/adversarial_loop.py",
            "--cycle",      str(cycle),
            "--max-cycles", str(MAX_CYCLES),
        ], "VALIDATE")

        time.sleep(2)

    else:
        final = get_accepted_count()
        print(f"\n  Reached MAX_CYCLES ({MAX_CYCLES}). "
              f"Final accepted count: {final}/{TARGET}.")

    print(f"\n  Dataset saved to {DATASET_CSV}")


if __name__ == "__main__":
    main()