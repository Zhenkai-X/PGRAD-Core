# -*- coding: utf-8 -*-
"""PGRAD gated reasoning.

Gate OFF retains the prior label without LLM label modification. Gate ON enters
constrained evidence review with the allowed states KEEP, SWITCH, and ABSTAIN.
ABSTAIN keeps the prior label for performance evaluation and marks manual review.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


VALID_CLASSES = ["NC", "PRISm", "COPD"]
TOP_K_EVIDENCE = 30
SUPPORT_TOPN = 3
CONFUSE_TOPM = 3
PER_CASE_EVIDENCE = 5

LAMBDA_DIRECTION = 0.4
LAMBDA_PERCENTILE = 0.3
LAMBDA_Z = 0.3

TAU_STRONG = 0.5067
TAU_WEAK = 0.1122
TAU_SIM = 1.6948

REVIEW_STATES = {"KEEP", "SWITCH", "ABSTAIN"}
_Z_RE = re.compile(r"z\s*=\s*([+-]?\d+(\.\d+)?)", re.I)
_P_RE = re.compile(r"P\s*([0-9]{1,3})")

Reviewer = Callable[[Dict[str, Any]], Dict[str, Any]]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def normalize_label(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return {0: "NC", 1: "PRISm", 2: "COPD"}.get(value, str(value))
    if isinstance(value, float) and value.is_integer():
        return normalize_label(int(value))
    text = str(value).strip()
    low = text.lower()
    if low in {"nc", "normal", "nl", "healthy", "control", "0"}:
        return "NC"
    if low in {"prism", "pris", "pr", "1"}:
        return "PRISm"
    if low in {"copd", "2"}:
        return "COPD"
    return text


def load_jsonl(path: str | Path) -> List[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_prob_dict(prob: Any) -> Dict[str, float]:
    if not isinstance(prob, dict):
        return {}
    out = {cls: 0.0 for cls in VALID_CLASSES}
    found = False
    for key, value in prob.items():
        label = normalize_label(key)
        if label in out:
            val = safe_float(value, -1.0)
            if val >= 0:
                out[label] = val
                found = True
    total = sum(out.values())
    if not found or total <= 0:
        return {}
    return {cls: out[cls] / total for cls in VALID_CLASSES}


def sorted_top2(prob: Dict[str, Any]) -> Tuple[str, str, float, float]:
    norm = normalize_prob_dict(prob)
    items = sorted(norm.items(), key=lambda item: item[1], reverse=True)
    if not items:
        return "", "", 0.0, 0.0
    if len(items) == 1:
        return items[0][0], "", items[0][1], 0.0
    return items[0][0], items[1][0], items[0][1], items[1][1]


def prior_label(prob: Dict[str, Any], fallback: Any = "") -> str:
    label = normalize_label(fallback)
    if label in VALID_CLASSES:
        return label
    top1, _, _, _ = sorted_top2(prob)
    return top1


def margin(prob: Dict[str, Any]) -> float:
    _, _, p1, p2 = sorted_top2(prob)
    return float(p1 - p2)


def parse_evidence_items(obj: dict) -> List[Dict[str, Any]]:
    items = obj.get("evidence_items")
    if isinstance(items, list) and all(isinstance(item, dict) for item in items):
        parsed = []
        for item in items:
            feat = str(item.get("feature", "")).strip()
            if not feat:
                continue
            score = item.get("score")
            z_value = item.get("z")
            parsed.append(
                {
                    "feature": feat,
                    "text": str(item.get("text", "")),
                    "z": safe_float(z_value, 0.0)
                    if isinstance(z_value, (int, float))
                    else z_value,
                    "percentile": item.get("percentile"),
                    "score": safe_float(score, 1.0),
                }
            )
        return topk_evidence(parsed, TOP_K_EVIDENCE)

    parsed = []
    for row in obj.get("knowledge_graph") or []:
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            continue
        text = str(row[1])
        feat = str(row[2]).strip()
        if not feat:
            continue
        z_match = _Z_RE.search(text)
        p_match = _P_RE.search(text)
        z_value = safe_float(z_match.group(1), 0.0) if z_match else 0.0
        pct = int(p_match.group(1)) if p_match else None
        parsed.append(
            {
                "feature": feat,
                "text": text,
                "z": z_value,
                "percentile": pct,
                "score": abs(z_value),
            }
        )
    return topk_evidence(parsed, TOP_K_EVIDENCE)


def topk_evidence(
    items: List[Dict[str, Any]], k: int = TOP_K_EVIDENCE
) -> List[Dict[str, Any]]:
    copied = []
    for item in items:
        row = dict(item)
        row["score"] = safe_float(row.get("score"), 1.0)
        copied.append(row)
    copied.sort(key=lambda item: item["score"], reverse=True)
    return copied[:k]


def evidence_direction(item: Dict[str, Any]) -> int:
    z_value = item.get("z")
    if isinstance(z_value, (int, float)):
        if abs(float(z_value)) < 1e-9:
            return 0
        return 1 if float(z_value) > 0 else -1
    text = str(item.get("text", "")).lower()
    if "high" in text:
        return 1
    if "low" in text:
        return -1
    return 0


def evidence_similarity(
    query: List[Dict[str, Any]], candidate: List[Dict[str, Any]]
) -> float:
    if not query or not candidate:
        return 0.0
    q_map = {item["feature"]: item for item in query if item.get("feature")}
    c_map = {item["feature"]: item for item in candidate if item.get("feature")}
    overlap = set(q_map) & set(c_map)
    denom = sum(safe_float(q_map[feat].get("score"), 1.0) for feat in q_map) or 1.0
    total = 0.0
    for feat in overlap:
        q = q_map[feat]
        c = c_map[feat]
        weight = safe_float(q.get("score"), 1.0)
        direction_score = (
            1.0 if evidence_direction(q) == evidence_direction(c) else 0.0
        )
        pq, pc = q.get("percentile"), c.get("percentile")
        pct_score = 0.0
        if isinstance(pq, int) and isinstance(pc, int):
            pct_score = 1.0 - min(abs(pq - pc), 100) / 100.0
        zq, zc = q.get("z"), c.get("z")
        z_score = 0.0
        if isinstance(zq, (int, float)) and isinstance(zc, (int, float)):
            z_score = 1.0 / (1.0 + abs(float(zq) - float(zc)))
        total += weight * (
            1.0
            + LAMBDA_DIRECTION * direction_score
            + LAMBDA_PERCENTILE * pct_score
            + LAMBDA_Z * z_score
        )
    return float(total / denom)


def matched_evidence(
    query: List[Dict[str, Any]], candidate: List[Dict[str, Any]]
) -> List[str]:
    c_map = {item["feature"]: item for item in candidate if item.get("feature")}
    out = []
    for item in query:
        feat = item.get("feature")
        if feat in c_map:
            out.append(str(c_map[feat].get("text", feat)))
        if len(out) >= PER_CASE_EVIDENCE:
            break
    return out


def retrieve_cases(
    pred_base: str,
    query_evidence: List[Dict[str, Any]],
    correct_db: Iterable[dict],
    error_db: Iterable[dict],
) -> Tuple[List[dict], List[dict]]:
    pred_base = normalize_label(pred_base)
    support_pool = []
    for obj in correct_db:
        if (
            normalize_label(obj.get("true_label")) == pred_base
            and normalize_label(obj.get("pred_base")) == pred_base
        ):
            ev = parse_evidence_items(obj)
            sim = evidence_similarity(query_evidence, ev)
            if sim > 0:
                support_pool.append((sim, obj, ev))

    confuse_pool = []
    for obj in error_db:
        if (
            normalize_label(obj.get("pred_base")) == pred_base
            and normalize_label(obj.get("true_label")) != pred_base
        ):
            ev = parse_evidence_items(obj)
            sim = evidence_similarity(query_evidence, ev)
            if sim > 0:
                confuse_pool.append((sim, obj, ev))

    support_pool.sort(key=lambda row: row[0], reverse=True)
    confuse_pool.sort(key=lambda row: row[0], reverse=True)

    support = [
        {
            "patient_id": str(obj.get("patient_id", "")),
            "true_label": normalize_label(obj.get("true_label")),
            "pred_base": normalize_label(obj.get("pred_base")),
            "sim": float(sim),
            "matched_evidence": matched_evidence(query_evidence, ev),
        }
        for sim, obj, ev in support_pool[:SUPPORT_TOPN]
    ]
    confuse = [
        {
            "patient_id": str(obj.get("patient_id", "")),
            "true_label": normalize_label(obj.get("true_label")),
            "pred_base": normalize_label(obj.get("pred_base")),
            "sim": float(sim),
            "matched_evidence": matched_evidence(query_evidence, ev),
        }
        for sim, obj, ev in confuse_pool[:CONFUSE_TOPM]
    ]
    return support, confuse


def gate_decision(
    prior_margin: float, max_confusable_similarity: float
) -> Tuple[bool, List[str]]:
    if prior_margin >= TAU_STRONG and max_confusable_similarity < TAU_SIM:
        return False, ["margin >= tau_strong and max confusable similarity < tau_sim"]
    if prior_margin <= TAU_WEAK:
        return True, ["margin <= tau_weak"]
    if max_confusable_similarity >= TAU_SIM:
        return True, ["max confusable similarity >= tau_sim"]
    return False, ["intermediate margin without high-similarity confusable case"]


def parse_review(review: Optional[Dict[str, Any]], pred_base: str) -> Dict[str, Any]:
    if not isinstance(review, dict):
        return {
            "state": "ABSTAIN",
            "switch_label": pred_base,
            "rationale": "No constrained review was returned.",
        }
    state = str(review.get("state", "")).strip().upper()
    if state not in REVIEW_STATES:
        state = "ABSTAIN"
    switch_label = normalize_label(review.get("switch_label", ""))
    if switch_label not in VALID_CLASSES:
        switch_label = pred_base
    return {
        "state": state,
        "switch_label": switch_label,
        "rationale": str(review.get("rationale", "")).strip(),
    }


def apply_review(
    pred_base: str, gate_on: bool, review: Optional[Dict[str, Any]]
) -> Tuple[str, str, bool]:
    if not gate_on:
        return "KEEP", pred_base, False
    parsed = parse_review(review, pred_base)
    state = parsed["state"]
    if state == "KEEP":
        return "KEEP", pred_base, False
    if state == "SWITCH" and parsed["switch_label"] != pred_base:
        return "SWITCH", parsed["switch_label"], False
    return "ABSTAIN", pred_base, True


def build_review_payload(
    obj: dict,
    pred_base: str,
    prob: Dict[str, float],
    query_evidence: List[Dict[str, Any]],
    support: List[dict],
    confuse: List[dict],
    prior_margin: float,
) -> Dict[str, Any]:
    return {
        "patient_id": obj.get("patient_id", ""),
        "allowed_states": sorted(REVIEW_STATES),
        "instruction": "Return KEEP, SWITCH, or ABSTAIN using only current evidence and retrieved cases.",
        "prior_label": pred_base,
        "prior_probability": prob,
        "margin": prior_margin,
        "current_evidence": query_evidence,
        "supportive_cases": support,
        "confusable_cases": confuse,
    }


def llm_evidence_review(
    client: Any,
    payload: Dict[str, Any],
    model: str,
    temperature: float = 0.1,
) -> Dict[str, Any]:
    """Ask the LLM for a constrained review state.

    The returned state is still validated by ``parse_review`` and can never
    bypass the KEEP/SWITCH/ABSTAIN rules in this module.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a constrained evidence reviewer. Return JSON only with "
                "state, switch_label, and rationale. state must be KEEP, SWITCH, "
                "or ABSTAIN. Use only the supplied evidence and retrieved cases."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    content = response.choices[0].message.content or "{}"
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.S)
        return json.loads(match.group(0)) if match else {}


def structured_report(
    result: Dict[str, Any],
    client: Any = None,
    model: Optional[str] = None,
    temperature: float = 0.1,
) -> Dict[str, Any]:
    """Return a structured report without allowing report text to alter labels."""
    report = {
        "patient_id": result.get("patient_id", ""),
        "prior_label": result.get("prior_label", ""),
        "final_label": result.get("final_label", ""),
        "review_state": result.get("review_state", "ABSTAIN"),
        "manual_review": bool(result.get("manual_review", False)),
        "gate_on": bool(result.get("gate_on", False)),
        "margin": result.get("margin", 0.0),
        "max_confusable_similarity": result.get("max_confusable_similarity", 0.0),
        "supportive_cases": result.get("supportive_cases", []),
        "confusable_cases": result.get("confusable_cases", []),
    }
    if client is None or not model:
        return report

    prompt = {
        "instruction": (
            "Generate a concise structured report from the supplied PGRAD result. "
            "Do not change prior_label, final_label, review_state, or manual_review. "
            "Return JSON with summary, key_evidence, supportive_cases, "
            "confusable_cases, and manual_review."
        ),
        "result": report,
    }
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt["instruction"]},
            {"role": "user", "content": json.dumps(prompt["result"], ensure_ascii=False)},
        ],
        temperature=temperature,
    )
    content = response.choices[0].message.content or "{}"
    try:
        llm_report = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.S)
        llm_report = json.loads(match.group(0)) if match else {}
    if not isinstance(llm_report, dict):
        return report
    report["summary"] = str(llm_report.get("summary", "")).strip()
    report["key_evidence"] = llm_report.get("key_evidence", [])
    return report


def reason_case(
    obj: dict,
    correct_db: List[dict],
    error_db: List[dict],
    reviewer: Optional[Reviewer] = None,
) -> Dict[str, Any]:
    prob = normalize_prob_dict(obj.get("prob"))
    pred_base = prior_label(prob, obj.get("pred_base"))
    query_evidence = parse_evidence_items(obj)
    support, confuse = retrieve_cases(pred_base, query_evidence, correct_db, error_db)
    max_confuse = max(
        [safe_float(row.get("sim"), 0.0) for row in confuse], default=0.0
    )
    prior_margin = margin(prob)
    gate_on, gate_reasons = gate_decision(prior_margin, max_confuse)

    review_payload = None
    review_response = None
    if gate_on:
        review_payload = build_review_payload(
            obj,
            pred_base,
            prob,
            query_evidence,
            support,
            confuse,
            prior_margin,
        )
        review_response = reviewer(review_payload) if reviewer else None

    state, final_label, manual_review = apply_review(
        pred_base, gate_on, review_response
    )
    return {
        "patient_id": obj.get("patient_id", ""),
        "prior_label": pred_base,
        "final_label": final_label,
        "review_state": state,
        "manual_review": bool(manual_review),
        "gate_on": bool(gate_on),
        "gate_reasons": gate_reasons,
        "margin": float(prior_margin),
        "max_confusable_similarity": float(max_confuse),
        "prob": prob,
        "supportive_cases": support,
        "confusable_cases": confuse,
        "review_payload": review_payload,
    }


def reason_dataset(
    evidence: List[dict], correct_db: List[dict], error_db: List[dict]
) -> List[Dict[str, Any]]:
    return [reason_case(obj, correct_db, error_db) for obj in evidence]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PGRAD gated reasoning.")
    parser.add_argument("--evidence", required=True, help="Current-case evidence JSONL.")
    parser.add_argument("--correct-db", required=True, help="Supportive evidence JSONL.")
    parser.add_argument("--error-db", required=True, help="Confusable evidence JSONL.")
    parser.add_argument("--out-jsonl", required=True)
    args = parser.parse_args()

    rows = reason_dataset(
        load_jsonl(args.evidence),
        load_jsonl(args.correct_db),
        load_jsonl(args.error_db),
    )
    with Path(args.out_jsonl).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"n_cases": len(rows), "out_jsonl": args.out_jsonl}, indent=2))


if __name__ == "__main__":
    main()
