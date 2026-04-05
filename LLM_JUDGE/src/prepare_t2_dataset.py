import re
from pathlib import Path

import pandas as pd
import sqlglot


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "generated_payloads.csv"
OUTPUT_PATH = BASE_DIR / "data" / "judge_samples_from_t2.csv"

context_map = {
    "login": "login",
    "search": "search",
    "user_lookup": "id_lookup",
    "order_filter": "product_filter",
    "comment_insert": "profile_update",
}


def safe_parse_sql(query: str) -> bool:
    try:
        sqlglot.parse_one(str(query))
        return True
    except Exception:
        return False


def normalize_text(text: str) -> str:
    return str(text).strip().lower()


def payload_inside_quotes_only(payload: str, full_query: str) -> bool:
    """
    Detects whether the payload appears to be fully trapped inside a quoted string
    rather than breaking out into executable SQL structure.
    """
    payload = str(payload)
    full_query = str(full_query)

    # simple quoted containment checks
    patterns = [
        f"'{payload}'",
        f'"{payload}"',
        f"%{payload}%",
    ]
    lowered = full_query.lower()
    for p in patterns:
        if p.lower() in lowered:
            return True

    # common LIKE pattern where payload remains inside string
    if f"like '%{payload.lower()}%'" in lowered:
        return True

    return False


def has_sqli_signal(payload: str, full_query: str, attack_category: str) -> bool:
    payload_l = normalize_text(payload)
    query_l = normalize_text(full_query)
    category_l = normalize_text(attack_category)

    generic_markers = [
        "' or ",
        '" or ',
        "' and ",
        '" and ',
        " union select",
        " union all select",
        "--",
        "/*",
        "*/",
        ";",
        "sleep(",
        "benchmark(",
        "information_schema",
        "drop table",
        "xp_cmdshell",
    ]

    category_markers = {
        "tautology": [" or 1=1", " and 1=1", "' or '1'='1", '" or "1"="1'],
        "union": [" union select", " union all select"],
        "blind": [" and ", " or ", "sleep(", "benchmark(", "case when", "if("],
        "boolean": [" and ", " or ", "1=1", "1=2"],
        "stacked": [";"],
        "comment": ["--", "/*", "*/", "#"],
    }

    # generic signal
    if any(m in payload_l for m in generic_markers):
        return True

    # category-aware signal
    for key, markers in category_markers.items():
        if key in category_l and any(m in payload_l or m in query_l for m in markers):
            return True

    return False


def derive_sandbox_malicious(payload: str, full_query: str, attack_category: str, syntax_valid: bool) -> bool:
    if not syntax_valid:
        return False

    # if the payload is only trapped inside quotes, treat it as not effectively malicious
    if payload_inside_quotes_only(payload, full_query):
        return False

    # require at least some SQLi signal
    if not has_sqli_signal(payload, full_query, attack_category):
        return False

    return True


def derive_notes(row: pd.Series, syntax_valid: bool, sandbox_malicious: bool) -> str:
    notes = [
        f"attack_category={row['attack_category']}",
        f"source={row['source']}",
        f"generator_model={row['generator_model']}",
    ]

    payload = normalize_text(row["payload"])
    full_query = normalize_text(row["full_query"])

    if payload_inside_quotes_only(row["payload"], row["full_query"]):
        notes.append("payload_inside_quotes_only")

    if not syntax_valid:
        notes.append("parse_failed")

    if not sandbox_malicious:
        notes.append("not_effectively_malicious")

    if payload == full_query:
        notes.append("same_as_full_query")

    return "; ".join(notes)


def main():
    df = pd.read_csv(INPUT_PATH)

    # Keep malicious rows if label exists
    if "label" in df.columns:
        df = df[df["label"] == 1].copy()

    df["context"] = df["template_context"].map(context_map).fillna("search")

    syntax_flags = []
    sandbox_flags = []
    notes_list = []

    for _, row in df.iterrows():
        full_query = str(row["full_query"])
        payload = str(row["payload"])
        attack_category = str(row["attack_category"])

        syntax_valid = safe_parse_sql(full_query)
        sandbox_malicious = derive_sandbox_malicious(
            payload=payload,
            full_query=full_query,
            attack_category=attack_category,
            syntax_valid=syntax_valid,
        )

        notes = derive_notes(row, syntax_valid, sandbox_malicious)

        syntax_flags.append(str(syntax_valid).lower())
        sandbox_flags.append(str(sandbox_malicious).lower())
        notes_list.append(notes)

    out = pd.DataFrame({
        "id": range(1, len(df) + 1),
        "seed_query": df["payload"].astype(str),
        "candidate_query": df["full_query"].astype(str),
        "context": df["context"].astype(str),
        "syntax_valid": syntax_flags,
        "sandbox_malicious": sandbox_flags,
        "notes": notes_list,
    })

    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(out)} rows to {OUTPUT_PATH}")

    print("\nDerived stats:")
    print("syntax_valid=true:", (out["syntax_valid"] == "true").sum())
    print("syntax_valid=false:", (out["syntax_valid"] == "false").sum())
    print("sandbox_malicious=true:", (out["sandbox_malicious"] == "true").sum())
    print("sandbox_malicious=false:", (out["sandbox_malicious"] == "false").sum())


if __name__ == "__main__":
    main()