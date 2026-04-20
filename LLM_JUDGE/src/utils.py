import json
import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests


BASE_DIR = Path(__file__).resolve().parent.parent
PROMPT_PATH = BASE_DIR / "prompts" / "judge_prompt.txt"
ALLOWED_FINAL_CONTEXTS = {"login", "search", "user_lookup", "order_filter", "comment_insert"}
LEGACY_CONTEXT_REMAP = {
    "id_lookup": "user_lookup",
    "product_filter": "order_filter",
    "profile_update": "comment_insert",
}

TRIVIAL_MUTATION_KEYWORDS = [
    "uppercase only",
    "lowercase only",
    "spacing only",
    "punctuation only",
    "same as seed",
    "cosmetic only",
    "case change only",
    "formatting-only rewrite",
    "case-only rewrite",
]

UNREALISTIC_OR_BROKEN_KEYWORDS = [
    "broken",
    "malformed",
    "unbalanced",
    "random tokens",
    "nonsense",
    "unrelated",
    "plain english",
    "natural language only",
    "payload_inside_quotes_only",
    "not_effectively_malicious",
    "parse_failed",
]


def load_system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _normalize(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _is_true_like(value: Any) -> bool:
    return _normalize(value) in {"true", "1", "yes", "y"}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return default


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _contains_any(text: Any, keywords: list[str]) -> bool:
    haystack = _normalize(text)
    return any(keyword in haystack for keyword in keywords)


def _normalize_query_id(raw: Any) -> str:
    raw_str = str(raw).strip()
    if not raw_str:
        return ""
    if raw_str.startswith("q"):
        return raw_str
    if raw_str.isdigit():
        return f"q{int(raw_str):05d}"
    return raw_str


def _to_json_list_text(value: Any) -> str:
    if value is None:
        return "[]"
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    text = str(value).strip()
    return text if text else "[]"


def _resolve_seed_payload(row: Dict[str, Any], payload: str) -> tuple[str, bool]:
    raw_seed = (
        row.get("seed_payload")
        or row.get("original_payload")
        or row.get("base_payload")
        or row.get("seed_query")
        or ""
    )
    seed_payload = str(raw_seed).strip()
    if not seed_payload:
        return "", False
    if _normalize(seed_payload) == _normalize(payload):
        return seed_payload, False
    return seed_payload, True


def parse_binary_label(value: Any, default: int = 1) -> int:
    text = _normalize(value)

    if text in {"1", "true", "yes", "y", "malicious", "attack", "sqli", "sql_injection", "positive"}:
        return 1
    if text in {"0", "false", "no", "n", "benign", "safe", "normal", "negative"}:
        return 0

    try:
        return int(float(text))
    except Exception:
        return default


def normalize_candidate_row(row: Dict[str, Any]) -> Dict[str, Any]:
    query_id = (
        row.get("query_id")
        or row.get("id")
        or row.get("sample_id")
        or row.get("row_id")
        or ""
    )

    payload = (
        row.get("payload")
        or row.get("candidate_payload")
        or row.get("mutated_payload")
        or row.get("candidate_query")
        or row.get("seed_query")
        or ""
    )
    payload = str(payload)

    seed_payload, seed_payload_available = _resolve_seed_payload(row, payload)

    full_query = (
        row.get("full_query")
        or row.get("candidate_full_query")
        or row.get("wrapped_query")
        or row.get("candidate_query")
        or ""
    )

    attack_category = row.get("llm_attack_category") or row.get("attack_category") or "unknown"
    template_context = row.get("template_context") or row.get("context") or "search"
    template_context = LEGACY_CONTEXT_REMAP.get(str(template_context).strip(), str(template_context).strip())

    raw_label = (
        row.get("label")
        if "label" in row
        else row.get("gold_label", row.get("class_label", row.get("is_malicious", 1)))
    )

    normalized = {
        "query_id": _normalize_query_id(query_id),
        "seed_payload": str(seed_payload),
        "seed_payload_available": seed_payload_available,
        "payload": payload,
        "full_query": str(full_query),
        "llm_attack_category": str(attack_category),
        "template_context": str(template_context),
        "label": parse_binary_label(raw_label, default=1),
        "mutation_count": _safe_int(row.get("mutation_count", 1), default=1),
        "codebert_score": _safe_float(row.get("codebert_score", 0.0), default=0.0),
        "ast_is_valid": _is_true_like(row.get("ast_is_valid", row.get("syntax_valid", True))),
        "ast_dialect": str(row.get("ast_dialect", "default")),
        "ast_node_set": _to_json_list_text(row.get("ast_node_set", "[]")),
        "sandbox_executed": _is_true_like(row.get("sandbox_executed", row.get("sandbox_malicious", True))),
        "sandbox_detection_mode": str(row.get("sandbox_detection_mode", "")),
        "sandbox_exploit_type": str(row.get("sandbox_exploit_type", "")),
        "sandbox_source": str(row.get("sandbox_source", "")),
        "notes": str(row.get("notes", "")),
        "existing_failure_stage": str(row.get("failure_stage", "")).strip(),
        "existing_failure_reason": str(row.get("failure_reason", "")).strip(),
    }

    if not normalized["query_id"]:
        normalized["query_id"] = (
            f"q_auto_{abs(hash((normalized['payload'], normalized['full_query']))) % 10**8:08d}"
        )

    return normalized


def build_user_prompt(row: Dict[str, Any]) -> str:
    seed_text = row["seed_payload"] if row.get("seed_payload_available") else "NOT PROVIDED"
    return f"""
Seed payload:
{seed_text}

Candidate payload:
{row['payload']}

Candidate full query:
{row['full_query']}

Attack category:
{row['llm_attack_category']}

Template context:
{row['template_context']}

AST valid:
{row['ast_is_valid']}

Sandbox executed successfully:
{row['sandbox_executed']}

Sandbox detection mode:
{row['sandbox_detection_mode']}

Sandbox exploit type:
{row['sandbox_exploit_type']}

Optional notes:
{row.get('notes', '')}

Return exactly one JSON object that matches the required schema.
Do not return markdown.
Do not return an empty object.
""".strip()


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()

    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("Model response was not a JSON object.")
        return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model response: {text[:400]}")

    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Extracted JSON was not an object.")
    return parsed


def _seed_similarity(row: Dict[str, Any]) -> Optional[float]:
    if not row.get("seed_payload_available"):
        return None
    return _similarity(row["seed_payload"], row["payload"])


def heuristic_judge(row: Dict[str, Any], similarity_threshold: float = 0.93) -> Tuple[Dict[str, Any], str]:
    seed_payload = row.get("seed_payload", "")
    payload = row["payload"]
    full_query = row["full_query"]
    context = _normalize(row["template_context"])
    syntax_ok = bool(row["ast_is_valid"])
    sandbox_ok = bool(row["sandbox_executed"])
    notes = row.get("notes", "")
    joined = _normalize(f"{seed_payload} || {payload} || {full_query} || {notes}")
    sim = _seed_similarity(row)

    malicious_intent_preserved = syntax_ok and sandbox_ok and not _contains_any(
        joined,
        [
            "benign",
            "harmless",
            "unrelated",
            "nonsense",
            "plain english",
            "natural language only",
            "payload_inside_quotes_only",
            "not_effectively_malicious",
            "parse_failed",
        ],
    )

    realistic_for_context = context in ALLOWED_FINAL_CONTEXTS and not _contains_any(
        joined,
        UNREALISTIC_OR_BROKEN_KEYWORDS,
    )

    non_trivial_mutation = True
    if sim is not None and sim >= similarity_threshold:
        non_trivial_mutation = False
    if _contains_any(joined, TRIVIAL_MUTATION_KEYWORDS):
        non_trivial_mutation = False

    score = 5
    if not malicious_intent_preserved:
        score -= 2
    if not realistic_for_context:
        score -= 1
    if not non_trivial_mutation:
        score -= 1
    score = max(1, min(5, score))

    keep = malicious_intent_preserved and realistic_for_context and non_trivial_mutation and score >= 4

    reason_parts = []
    if malicious_intent_preserved:
        reason_parts.append("malicious intent appears preserved")
    else:
        reason_parts.append("malicious intent is not clearly preserved")

    if realistic_for_context:
        reason_parts.append(f"candidate is plausible for {context}")
    else:
        reason_parts.append(f"candidate is weak or unrealistic for {context}")

    if non_trivial_mutation:
        if sim is None:
            reason_parts.append("mutation appears non-trivial based on content and notes")
        else:
            reason_parts.append("mutation is meaningfully different from the seed")
    else:
        reason_parts.append("mutation looks mostly cosmetic")

    result = {
        "malicious_intent_preserved": malicious_intent_preserved,
        "realistic_for_context": realistic_for_context,
        "non_trivial_mutation": non_trivial_mutation,
        "overall_quality_score": score,
        "keep": keep,
        "reason": "; ".join(reason_parts) + ".",
    }
    return result, "heuristic"


def ollama_local_generate(system_prompt: str, user_prompt: str) -> str:
    model = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")
    url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

    judge_schema = {
        "type": "object",
        "properties": {
            "malicious_intent_preserved": {"type": "boolean"},
            "realistic_for_context": {"type": "boolean"},
            "non_trivial_mutation": {"type": "boolean"},
            "overall_quality_score": {"type": "integer", "minimum": 1, "maximum": 5},
            "keep": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": [
            "malicious_intent_preserved",
            "realistic_for_context",
            "non_trivial_mutation",
            "overall_quality_score",
            "keep",
            "reason",
        ],
    }

    grounded_prompt = (
        system_prompt
        + "\n\nYou must return exactly one JSON object matching this schema:\n"
        + json.dumps(judge_schema, indent=2)
        + "\n\n"
        + user_prompt
    )

    request_payload = {
        "model": model,
        "system": "You are a strict JSON generator. Return only one JSON object.",
        "prompt": grounded_prompt,
        "format": judge_schema,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 220},
    }

    response = requests.post(url, json=request_payload, timeout=240)
    response.raise_for_status()
    data = response.json()

    if "response" not in data:
        raise ValueError(f"Unexpected Ollama response: {data}")

    text = str(data["response"]).strip()
    if not text:
        raise ValueError(f"Empty Ollama response. Full payload: {data}")

    return text


def apply_hard_rules(
    row: Dict[str, Any],
    parsed: Dict[str, Any],
    similarity_threshold: float = 0.93,
) -> Dict[str, Any]:
    notes = _normalize(row.get("notes", ""))
    context = _normalize(row["template_context"])
    syntax_ok = bool(row["ast_is_valid"])
    sandbox_ok = bool(row["sandbox_executed"])
    sim = _seed_similarity(row)

    result = dict(parsed)
    result.setdefault("malicious_intent_preserved", False)
    result.setdefault("realistic_for_context", False)
    result.setdefault("non_trivial_mutation", False)
    result.setdefault("overall_quality_score", 1)
    result.setdefault("keep", False)
    result.setdefault("reason", "No reason provided.")

    reason_parts = [str(result["reason"]).strip()]

    if not syntax_ok or not sandbox_ok:
        result["malicious_intent_preserved"] = False
        result["keep"] = False
        result["overall_quality_score"] = min(int(result["overall_quality_score"]), 2)
        reason_parts.append("Rejected by hard rule because AST validation or sandbox execution failed.")

    if _contains_any(notes, TRIVIAL_MUTATION_KEYWORDS):
        result["non_trivial_mutation"] = False
        result["keep"] = False
        result["overall_quality_score"] = min(int(result["overall_quality_score"]), 2)
        reason_parts.append("Rejected by hard rule because notes indicate a trivial mutation.")

    if sim is not None and sim >= similarity_threshold:
        result["non_trivial_mutation"] = False
        result["keep"] = False
        result["overall_quality_score"] = min(int(result["overall_quality_score"]), 2)
        reason_parts.append(
            f"Rejected by hard rule because seed/candidate similarity is too high "
            f"({sim:.2f} >= {similarity_threshold:.2f})."
        )

    if _contains_any(notes, UNREALISTIC_OR_BROKEN_KEYWORDS):
        result["realistic_for_context"] = False
        result["keep"] = False
        result["overall_quality_score"] = min(int(result["overall_quality_score"]), 2)
        reason_parts.append("Rejected by hard rule because notes indicate unrealistic or broken content.")

    if context not in ALLOWED_FINAL_CONTEXTS:
        result["realistic_for_context"] = False
        result["keep"] = False
        result["overall_quality_score"] = min(int(result["overall_quality_score"]), 2)
        reason_parts.append("Rejected by hard rule because context is outside the allowed task contexts.")

    result["keep"] = (
        bool(result["malicious_intent_preserved"])
        and bool(result["realistic_for_context"])
        and bool(result["non_trivial_mutation"])
        and int(result["overall_quality_score"]) >= 4
    )
    result["reason"] = " ".join(part for part in reason_parts if part)
    return result


def sandbox_failure_reason(row: Dict[str, Any]) -> str:
    notes = _normalize(row.get("notes", ""))
    if "parse_failed" in notes or not row["ast_is_valid"]:
        return "Sandbox returned benign. Reason: AST validation failed or query did not parse cleanly"
    if "payload_inside_quotes_only" in notes:
        return "Sandbox returned benign. Reason: payload stayed neutralized inside the quoted template"
    if "not_effectively_malicious" in notes:
        return "Sandbox returned benign. Reason: no exploit detected in template execution"
    return "Sandbox returned benign. Reason: no exploit detected"


def rejection_hint(stage: str) -> str:
    if stage == "sandbox":
        return "The query was classified as benign by the sandbox. Try a different injection technique."
    return "The LLM judge rejected this query. Make the payload more realistic for its SQL context and ensure the malicious intent is clearly preserved."