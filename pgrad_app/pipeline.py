# -*- coding: utf-8 -*-
from __future__ import annotations

import io
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency when LLM is disabled.
    OpenAI = None


APP_DIR = Path(__file__).resolve().parents[1]

VALID_CLASSES = ["NC", "PRISm", "COPD"]
ID_COL_CANDIDATES = ["ID", "patient_id", "PatientID", "subject_id", "RID", "PTID", "id"]
LABEL_COL_CANDIDATES = ["label", "true_label", "diagnosis", "group", "class", "Label"]

REF_CLASS = "NC"
TOP_K_EVIDENCE = 12
QUERY_TOPK_EVID = 15
SUPPORT_TOPN = 10
CONFUSE_TOPM = 10
PER_CASE_EVID = 5
Z_NEAR = 0.25

MARGIN_STRONG = 0.5067
MARGIN_WEAK = 0.1122
SIM_TRIGGER = 1.6948
DELTA_SWITCH = 0.100
DELTA_KEEP = 0.040
SIM_SWITCH_MIN = 1.480

MODEL_NAME = "qwen3.6-plus"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
TEMP_PROB_UPDATE = 0.0
TEMP_REPORT = 0.2
NETWORK_MAX_RETRIES = 3
NETWORK_RETRY_SECONDS = 2.0

_Z_RE = re.compile(r"z\s*=\s*([+-]?\d+(\.\d+)?)", re.I)
_P_RE = re.compile(r"P\s*([0-9]{1,3})")
_TAG_RE = re.compile(r"\b([ESF][1-9]\d?)\b")
_JSON_OBJ_RE = re.compile(r"(\{.*\})", re.S)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")

ProgressCallback = Callable[[str, str], None]


@dataclass
class RefStats:
    ref: Dict[str, Dict[str, object]] = field(default_factory=dict)
    mean_by_class: Dict[str, Dict[str, float]] = field(default_factory=dict)


def normalize_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        if re.fullmatch(r"\d+(\.0+)?", text):
            return str(int(float(text)))
    except Exception:
        pass
    return text


def normalize_label(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return {0: "NC", 1: "PRISm", 2: "COPD"}.get(int(value), str(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return normalize_label(int(value))
    text = str(value).strip()
    low = text.lower()
    if low in ["nc", "normal", "nl", "healthy", "control", "0"]:
        return "NC"
    if low in ["prism", "pris", "pr", "1"]:
        return "PRISm"
    if low in ["copd", "2"]:
        return "COPD"
    return text if text in VALID_CLASSES else text


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def fmt_value(value: float) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "NA"
    absolute = abs(float(value))
    if absolute >= 1000 or (0 < absolute < 1e-3):
        return f"{float(value):.6g}"
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def read_any_table_from_upload(uploaded_file: Any) -> pd.DataFrame:
    name = getattr(uploaded_file, "name", "")
    suffix = Path(name).suffix.lower()
    data = uploaded_file.getvalue()
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(io.BytesIO(data), engine="openpyxl")
    for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"]:
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            continue
    return pd.read_csv(io.BytesIO(data), engine="python")


def read_any_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"]:
            try:
                return pd.read_csv(path, encoding=enc)
            except Exception:
                continue
        return pd.read_csv(path, engine="python")
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path, engine="openpyxl")
    raise ValueError(f"Unsupported table type: {suffix}")


def detect_id_col(df: pd.DataFrame, preferred: Optional[str] = None) -> str:
    if preferred and preferred in df.columns:
        return preferred
    for col in ID_COL_CANDIDATES:
        if col in df.columns:
            return col
    for col in df.columns:
        if "id" in str(col).lower():
            return str(col)
    df["patient_id"] = [f"case_{i + 1}" for i in range(len(df))]
    return "patient_id"


def detect_label_col(df: pd.DataFrame, preferred: Optional[str] = None) -> Optional[str]:
    if preferred and preferred in df.columns:
        return preferred
    for col in LABEL_COL_CANDIDATES:
        if col in df.columns:
            return col
    return None


def empirical_percentile(
    sorted_arr: np.ndarray,
    value: float,
    percentiles: Optional[np.ndarray] = None,
    cumulative_counts: Optional[np.ndarray] = None,
    total_count: Optional[int] = None,
) -> int:
    if sorted_arr is None or len(sorted_arr) == 0:
        return 50
    if cumulative_counts is not None and total_count:
        arr = np.asarray(sorted_arr, dtype=float)
        counts = np.asarray(cumulative_counts, dtype=float)
        idx = np.searchsorted(arr, value, side="right")
        count = 0.0 if idx <= 0 else float(counts[min(idx - 1, len(counts) - 1)])
        return max(0, min(100, int(round(100.0 * count / float(total_count)))))
    if percentiles is not None and len(percentiles) == len(sorted_arr):
        arr = np.asarray(sorted_arr, dtype=float)
        pct_arr = np.asarray(percentiles, dtype=float)
        mask = np.isfinite(arr) & np.isfinite(pct_arr)
        if not np.any(mask):
            return 50
        arr = arr[mask]
        pct_arr = pct_arr[mask]
        order = np.argsort(arr)
        arr = arr[order]
        pct_arr = pct_arr[order]
        unique_vals, inverse = np.unique(arr, return_inverse=True)
        unique_pct = np.zeros(len(unique_vals), dtype=float)
        for idx, pct in zip(inverse, pct_arr):
            unique_pct[idx] = max(unique_pct[idx], pct)
        if len(unique_vals) == 1:
            return max(0, min(100, int(round(unique_pct[0]))))
        pct = np.interp(float(value), unique_vals, unique_pct, left=0.0, right=100.0)
        return max(0, min(100, int(round(float(pct)))))
    idx = np.searchsorted(sorted_arr, value, side="right")
    return max(0, min(100, int(round(100.0 * idx / len(sorted_arr)))))


def build_ref_stats(df: pd.DataFrame, feats: List[str], label_col: str) -> RefStats:
    stats = RefStats()
    if label_col not in df.columns:
        return stats
    tmp = df.copy()
    tmp[label_col] = tmp[label_col].apply(normalize_label)
    tmp = tmp[tmp[label_col].isin(VALID_CLASSES)].reset_index(drop=True)
    for feat in feats:
        if feat not in tmp.columns:
            continue
        stats.mean_by_class[feat] = {}
        for cls in VALID_CLASSES:
            vals = pd.to_numeric(tmp.loc[tmp[label_col] == cls, feat], errors="coerce").dropna()
            if len(vals) > 0:
                stats.mean_by_class[feat][cls] = float(np.mean(vals.to_numpy(dtype=float)))
    ref = tmp[tmp[label_col] == REF_CLASS].copy()
    for feat in feats:
        if feat not in ref.columns:
            continue
        vals = pd.to_numeric(ref[feat], errors="coerce").dropna()
        if len(vals) < 3:
            continue
        arr = vals.to_numpy(dtype=float)
        stats.ref[feat] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr) + 1e-12),
            "sorted": np.sort(arr),
        }
    return stats


def load_ref_stats(path: Path) -> RefStats:
    if not path.exists():
        raise RuntimeError(f"Reference statistics file was not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    stats = RefStats()
    for feat, item in (payload.get("features") or {}).items():
        ref = item.get("ref") or {}
        if {"mean", "std", "cdf_values", "cdf_counts", "count"} <= set(ref):
            stats.ref[feat] = {
                "mean": float(ref["mean"]),
                "std": float(ref["std"]),
                "sorted": np.asarray(ref["cdf_values"], dtype=float),
                "cumulative_counts": np.asarray(ref["cdf_counts"], dtype=float),
                "total_count": int(ref["count"]),
            }
        elif {"mean", "std", "quantiles", "quantile_percentiles"} <= set(ref):
            stats.ref[feat] = {
                "mean": float(ref["mean"]),
                "std": float(ref["std"]),
                "sorted": np.asarray(ref["quantiles"], dtype=float),
                "percentiles": np.asarray(ref["quantile_percentiles"], dtype=float),
            }
        means = item.get("mean_by_class") or {}
        stats.mean_by_class[feat] = {normalize_label(cls): float(val) for cls, val in means.items()}
    return stats


def z_and_percentile(ref_stats: RefStats, feat: str, value: float) -> Tuple[Optional[float], Optional[int]]:
    if feat not in ref_stats.ref:
        return None, None
    ref = ref_stats.ref[feat]
    z = (value - float(ref["mean"])) / (float(ref["std"]) + 1e-12)
    pct = empirical_percentile(
        ref["sorted"],
        value,
        ref.get("percentiles"),
        ref.get("cumulative_counts"),
        ref.get("total_count"),
    )
    return float(z), int(pct)


def direction_text(z_value: float) -> str:
    if abs(z_value) < Z_NEAR:
        return "near"
    return "high" if z_value > 0 else "low"


def support_text(ref_stats: RefStats, feat: str, z_value: float, pred_cls: str) -> str:
    pred_cls = normalize_label(pred_cls)
    if pred_cls not in VALID_CLASSES or pred_cls == REF_CLASS:
        return ""
    means = ref_stats.mean_by_class.get(feat, {})
    if REF_CLASS not in means or pred_cls not in means:
        return ""
    delta = float(means[pred_cls] - means[REF_CLASS])
    if abs(z_value) < Z_NEAR:
        return "neutral"
    if delta > 0:
        return "consistent" if z_value > 0 else "opposite"
    if delta < 0:
        return "consistent" if z_value < 0 else "opposite"
    return ""


def load_feature_weights(path: Path, feats: Iterable[str]) -> Dict[str, float]:
    weights = {feat: 1.0 for feat in feats}
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(path, encoding="gb18030")
    if "feature" not in df.columns:
        return weights
    score_col = "fused_score" if "fused_score" in df.columns else ("score" if "score" in df.columns else None)
    if not score_col:
        return weights
    for _, row in df.iterrows():
        feat = str(row["feature"])
        if feat in weights:
            weights[feat] = safe_float(row[score_col], 1.0)
    return weights


def bundle_features(bundle: Dict[str, Any]) -> List[str]:
    return list(bundle.get("selected_features") or bundle.get("features") or bundle.get("feature_names") or [])


def select_model_profile_for_upload(
    df_ext: pd.DataFrame,
    model_path: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    upload_cols = {str(col).strip() for col in df_ext.columns}
    if not model_path.exists():
        raise FileNotFoundError("Required model file was not provided.")
    bundle = joblib.load(model_path)
    feats = bundle_features(bundle)
    missing = [feat for feat in feats if feat not in upload_cols]
    profile = {
        "key": "configured_model",
        "display_name": "Configured PGRAD model",
        "model_path": model_path,
        "feature_count": len(feats),
        "features": feats,
    }
    return profile, bundle, missing


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def english_only_text(text: Any) -> str:
    out = "" if text is None else str(text)
    replacements = [
        ("\uff1b", "; "),
        ("\uff0c", ", "),
        ("\uff1a", ": "),
        ("\u65b9\u5411\u4e00\u81f4", " direction consistent"),
        ("\u65b9\u5411\u76f8\u53cd", " direction opposite"),
        ("\u76f8\u5bf9", "relative to "),
        ("\u504f\u9ad8", "high"),
        ("\u504f\u4f4e", "low"),
        ("\u63a5\u8fd1", "near"),
        ("\u7ea6", "approx "),
        ("\u4e0e", "with "),
        ("\u8bca\u65ad\u4e3a", "diagnosis is"),
    ]
    for old, new in replacements:
        out = out.replace(old, new)
    out = _CJK_RE.sub("", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"[ \t]+\n", "\n", out)
    return out.strip()


def parse_evidence_items(obj: dict) -> List[Dict[str, Any]]:
    ev = obj.get("evidence_items")
    if isinstance(ev, list) and ev and isinstance(ev[0], dict) and "feature" in ev[0]:
        out = []
        for item in ev:
            feat = str(item.get("feature", "")).strip()
            text = english_only_text(item.get("text", ""))
            z_value = item.get("z")
            pct = item.get("percentile")
            score = item.get("score")
            if not isinstance(score, (int, float)):
                score = abs(z_value) if isinstance(z_value, (int, float)) else 1.0
            out.append({"feature": feat, "text": text, "z": z_value, "percentile": pct, "score": float(score)})
        return out

    out = []
    kg = obj.get("knowledge_graph")
    if isinstance(kg, list):
        for row in kg:
            if not isinstance(row, (list, tuple)) or len(row) < 3:
                continue
            text = english_only_text(row[1])
            feat = str(row[2]).strip()
            z_value = None
            pct = None
            match_z = _Z_RE.search(text)
            if match_z:
                z_value = safe_float(match_z.group(1), 0.0)
            match_p = _P_RE.search(text)
            if match_p:
                pct = int(match_p.group(1))
            score = abs(z_value) if isinstance(z_value, (int, float)) else 1.0
            if feat:
                out.append({"feature": feat, "text": text, "z": z_value, "percentile": pct, "score": float(score)})
    return out


def topk_evidence(items: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    copied = []
    for item in items:
        item2 = dict(item)
        score = item2.get("score")
        if not isinstance(score, (int, float)):
            z_value = item2.get("z")
            score = abs(z_value) if isinstance(z_value, (int, float)) else 1.0
        item2["score"] = float(score)
        copied.append(item2)
    copied.sort(key=lambda item: item.get("score", 0.0), reverse=True)
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


def similarity(query: List[Dict[str, Any]], candidate: List[Dict[str, Any]]) -> float:
    if not query or not candidate:
        return 0.0
    q_map = {item["feature"]: item for item in query if item.get("feature")}
    c_map = {item["feature"]: item for item in candidate if item.get("feature")}
    overlap = set(q_map) & set(c_map)
    if not overlap:
        return 0.0
    denom = max(1.0, float(sum(item.get("score", 1.0) for item in query)))
    score = 0.0
    for feat in overlap:
        q = q_map[feat]
        c = c_map[feat]
        weight = float(q.get("score", 1.0))
        score += 1.0 * weight
        if evidence_direction(q) == evidence_direction(c):
            score += 0.4 * weight
        pq, pc = q.get("percentile"), c.get("percentile")
        if isinstance(pq, int) and isinstance(pc, int):
            score += (0.2 * weight) * (1.0 - min(abs(pq - pc), 100) / 100.0)
        zq, zc = q.get("z"), c.get("z")
        if isinstance(zq, (int, float)) and isinstance(zc, (int, float)):
            score += (0.3 * weight) * (1.0 / (1.0 + abs(float(zq) - float(zc))))
    return float(score) / denom


def pick_matched_evidence(query: List[Dict[str, Any]], candidate: List[Dict[str, Any]], max_items: int) -> List[str]:
    q_feats = [item["feature"] for item in query if item.get("feature")]
    c_map = {item["feature"]: item for item in candidate if item.get("feature")}
    out: List[str] = []
    for feat in q_feats:
        if feat in c_map:
            out.append(english_only_text(c_map[feat].get("text", feat)))
        if len(out) >= max_items:
            break
    return out


def retrieve_cases(query_pred: str, query_evid: List[Dict[str, Any]], correct_db: List[dict], error_db: List[dict]) -> Tuple[List[dict], List[dict]]:
    qpred = normalize_label(query_pred)
    support_pool = []
    for obj in correct_db:
        true_label = normalize_label(obj.get("true_label"))
        pred_base = normalize_label(obj.get("pred_base"))
        if true_label == qpred and pred_base == qpred:
            ev = topk_evidence(parse_evidence_items(obj), QUERY_TOPK_EVID)
            sim = similarity(query_evid, ev)
            if sim > 0:
                support_pool.append((sim, obj, ev))

    confuse_pool = []
    for obj in error_db:
        true_label = normalize_label(obj.get("true_label"))
        pred_base = normalize_label(obj.get("pred_base"))
        if pred_base == qpred and true_label != qpred:
            ev = topk_evidence(parse_evidence_items(obj), QUERY_TOPK_EVID)
            sim = similarity(query_evid, ev)
            if sim > 0:
                confuse_pool.append((sim, obj, ev))

    support_pool.sort(key=lambda row: row[0], reverse=True)
    confuse_pool.sort(key=lambda row: row[0], reverse=True)

    support = [
        {
            "patient_id": str(obj.get("patient_id", "")).strip(),
            "true_label": normalize_label(obj.get("true_label")),
            "pred_base": normalize_label(obj.get("pred_base")),
            "sim": float(sim),
            "matched_evidence": pick_matched_evidence(query_evid, ev, PER_CASE_EVID),
        }
        for sim, obj, ev in support_pool[:SUPPORT_TOPN]
    ]
    confuse = [
        {
            "patient_id": str(obj.get("patient_id", "")).strip(),
            "true_label": normalize_label(obj.get("true_label")),
            "pred_base": normalize_label(obj.get("pred_base")),
            "sim": float(sim),
            "matched_evidence": pick_matched_evidence(query_evid, ev, PER_CASE_EVID),
        }
        for sim, obj, ev in confuse_pool[:CONFUSE_TOPM]
    ]
    return support, confuse


def normalize_prob_dict(prob: Any) -> Dict[str, float]:
    if not isinstance(prob, dict):
        return {}
    out = {label: 0.0 for label in VALID_CLASSES}
    found = False
    for key, value in prob.items():
        label = normalize_label(key)
        if label not in out:
            continue
        val = safe_float(value, -1.0)
        if val < 0:
            continue
        out[label] = float(val)
        found = True
    total = sum(out.values())
    if not found or total <= 0:
        return {}
    return {label: out[label] / total for label in VALID_CLASSES}


def argmax_label(prob: Dict[str, Any]) -> str:
    norm = normalize_prob_dict(prob)
    if not norm:
        return ""
    return max(VALID_CLASSES, key=lambda label: norm.get(label, 0.0))


def format_prob(prob: Dict[str, Any]) -> str:
    pairs = []
    for key, value in (prob or {}).items():
        label = normalize_label(key)
        if label in VALID_CLASSES:
            pairs.append((label, safe_float(value, 0.0)))
    pairs.sort(key=lambda row: row[1], reverse=True)
    return "  ".join([f"{label}={value:.3f}" for label, value in pairs])


def sorted_top2(prob: Dict[str, Any]) -> Tuple[str, str, float, float]:
    items = [(normalize_label(k), safe_float(v, 0.0)) for k, v in (prob or {}).items()]
    items = [(k, v) for k, v in items if k in VALID_CLASSES]
    items.sort(key=lambda row: row[1], reverse=True)
    if not items:
        return "", "", 0.0, 0.0
    if len(items) == 1:
        return items[0][0], "", items[0][1], 0.0
    return items[0][0], items[1][0], items[0][1], items[1][1]


def top2_margin(prob: Dict[str, Any]) -> float:
    _, _, p1, p2 = sorted_top2(prob)
    return float(p1 - p2)


def precompute_risk(prob: Dict[str, Any], confuse: List[dict]) -> Tuple[str, str]:
    margin = top2_margin(prob)
    max_sim = max([safe_float(item.get("sim", 0.0), 0.0) for item in (confuse or [])], default=0.0)
    if margin >= MARGIN_STRONG and max_sim < SIM_TRIGGER:
        return "Low", "Large probability margin and no high-similarity error case was retrieved."
    if margin <= MARGIN_WEAK and max_sim >= SIM_TRIGGER:
        return "High", "Low probability margin with a high-similarity historical error case."
    if margin <= MARGIN_WEAK or max_sim >= SIM_TRIGGER:
        return "High", "One trigger condition is met: low confidence or high-similarity error pattern."
    return "Medium", "Borderline uncertainty is present but strong trigger conditions are not met."


def gate_trigger(margin: float, max_sim: float, risk_level: str) -> Tuple[bool, List[str]]:
    if margin >= MARGIN_STRONG and max_sim < SIM_TRIGGER and risk_level != "High":
        return False, ["Strong confidence and no high-similarity confusion case."]
    reasons = []
    if margin <= MARGIN_WEAK:
        reasons.append(f"Low confidence: margin={margin:.3f} <= {MARGIN_WEAK:.2f}.")
    if max_sim >= SIM_TRIGGER:
        reasons.append(f"High-risk confusion case: max_sim={max_sim:.3f} >= {SIM_TRIGGER:.2f}.")
    if risk_level == "High":
        reasons.append("Risk level is high.")
    if reasons:
        return True, reasons
    return False, [f"Intermediate confidence: {MARGIN_WEAK:.2f}<margin={margin:.3f}<{MARGIN_STRONG:.2f}; keep base model."]


def max_sim_and_case(cases: List[dict]) -> Tuple[float, Optional[dict]]:
    if not cases:
        return 0.0, None
    best = max(cases, key=lambda row: safe_float(row.get("sim", 0.0), 0.0))
    return safe_float(best.get("sim", 0.0), 0.0), best


def suggest_switch_label_from_confuse(confuse: List[dict], top1: str, top2: str, pred_base: str) -> Tuple[Optional[str], Optional[dict]]:
    pred_base = normalize_label(pred_base)
    allowed = [normalize_label(label) for label in [top1, top2] if normalize_label(label) and normalize_label(label) != pred_base]
    if not allowed:
        return None, None
    _, best_case = max_sim_and_case(confuse)
    if best_case:
        true_label = normalize_label(best_case.get("true_label", ""))
        if true_label in allowed:
            return true_label, best_case
    scores = {label: 0.0 for label in allowed}
    for row in confuse:
        true_label = normalize_label(row.get("true_label", ""))
        if true_label in scores:
            scores[true_label] += safe_float(row.get("sim", 0.0), 0.0)
    best_label = max(scores.items(), key=lambda row: row[1])[0] if scores else None
    if best_label and scores.get(best_label, 0.0) > 0:
        return best_label, best_case
    return None, best_case


def directional_decision(pred_base: str, prob: Dict[str, Any], margin: float, gate_on: bool, support: List[dict], confuse: List[dict]) -> Tuple[str, str, bool, List[str], str, Dict[str, Any]]:
    pred_base = normalize_label(pred_base)
    top1, top2, p1, p2 = sorted_top2(prob)
    s_correct, best_support = max_sim_and_case(support)
    s_confuse, best_confuse = max_sim_and_case(confuse)
    delta_sim = float(s_confuse - s_correct)
    switch_label, _ = suggest_switch_label_from_confuse(confuse, top1, top2, pred_base)
    stats = {
        "S_correct_max": float(s_correct),
        "S_confuse_max": float(s_confuse),
        "delta_sim": float(delta_sim),
        "best_support_case": best_support.get("patient_id") if best_support else "",
        "best_confuse_case": best_confuse.get("patient_id") if best_confuse else "",
        "switch_label_suggested": switch_label or "",
        "top2": {"top1": top1, "top2": top2, "p1": p1, "p2": p2},
    }
    if not gate_on:
        return "keep", pred_base, False, ["Gate was not triggered; the base model diagnosis is retained."], "", stats
    notes = f"S_confuse={s_confuse:.3f}, S_correct={s_correct:.3f}, delta={delta_sim:.3f}"
    if delta_sim >= DELTA_SWITCH and s_confuse >= SIM_SWITCH_MIN and switch_label:
        reasons = [f"Directional evidence supports switching: delta={delta_sim:.3f} >= {DELTA_SWITCH:.2f} and S_confuse={s_confuse:.3f} >= {SIM_SWITCH_MIN:.2f}."]
        if best_confuse:
            reasons.append(f"Nearest confusion case is {best_confuse.get('patient_id', '')}, true={normalize_label(best_confuse.get('true_label'))}.")
        return "switch", switch_label, False, reasons, notes, stats
    if delta_sim <= -DELTA_KEEP:
        reasons = [f"Directional evidence supports keeping the base label: delta={delta_sim:.3f} <= -{DELTA_KEEP:.2f}."]
        if best_support:
            reasons.append(f"Nearest correct support case is {best_support.get('patient_id', '')}.")
        return "keep", pred_base, False, reasons, notes, stats
    return "abstain", pred_base, True, [f"Directional evidence is insufficient: delta={delta_sim:.3f}; expert review is recommended."], notes, stats


def build_external_evidence(
    df_ext: pd.DataFrame,
    bundle: Dict[str, Any],
    ref_stats: RefStats,
    model_profile: Dict[str, Any],
    feature_scores_path: Path,
) -> Tuple[List[dict], pd.DataFrame, List[str]]:
    pipeline = bundle["pipeline"]
    selected_feats = bundle_features(bundle)
    class_names = [normalize_label(c) for c in list(bundle.get("class_names", VALID_CLASSES))]
    id_col = detect_id_col(df_ext, bundle.get("id_col"))

    df = df_ext.copy()
    df[id_col] = df[id_col].apply(normalize_id)
    missing = [feat for feat in selected_feats if feat not in df.columns]

    x_ext = df.reindex(columns=selected_feats).apply(pd.to_numeric, errors="coerce")
    proba_raw = pipeline.predict_proba(x_ext)
    probs = []
    for i in range(len(df)):
        probs.append(normalize_prob_dict({class_names[j]: float(proba_raw[i, j]) for j in range(len(class_names))}))
    pred_base = [argmax_label(prob) for prob in probs]

    feats_all = [feat for feat in selected_feats if feat in ref_stats.ref]
    weights = load_feature_weights(feature_scores_path, feats_all)
    external_db = []
    pred_rows = []

    for i in range(len(df)):
        pid = str(df.loc[i, id_col])
        pred_cls = normalize_label(pred_base[i])
        prob = probs[i]
        row = df.loc[i]
        candidates = []
        for feat in feats_all:
            value = pd.to_numeric(pd.Series([row.get(feat, np.nan)]), errors="coerce").iloc[0]
            if pd.isna(value):
                continue
            z_value, pct = z_and_percentile(ref_stats, feat, float(value))
            score = 0.0 if z_value is None else abs(float(z_value)) * float(weights.get(feat, 1.0))
            candidates.append((float(score), feat, float(value), z_value, pct))
        candidates.sort(key=lambda row2: row2[0], reverse=True)
        candidates = candidates[:TOP_K_EVIDENCE]

        kg = []
        evidence_items = []
        for score, feat, value, z_value, pct in candidates:
            value_str = fmt_value(value)
            if z_value is None or pct is None:
                text = f"{feat}={value_str}; missing NC reference distribution"
                z_out, pct_out = None, None
            else:
                text = f"{feat}={value_str}; relative to {REF_CLASS}: {direction_text(float(z_value))}; z={float(z_value):+0.2f}; approx P{int(pct)}"
                support = support_text(ref_stats, feat, float(z_value), pred_cls)
                if support == "consistent":
                    text += f"; consistent with {pred_cls}"
                elif support == "opposite":
                    text += f"; opposite to {pred_cls}"
                z_out, pct_out = float(z_value), int(pct)
            kg.append([value_str, text, feat])
            evidence_items.append({
                "feature": feat,
                "value": float(value),
                "value_str": value_str,
                "z": z_out,
                "percentile": pct_out,
                "score": float(score),
                "text": text,
            })

        kg.append([pid, f"Base model prediction is {pred_cls}", pred_cls])
        external_db.append({
            "patient_id": pid,
            "knowledge_graph": kg,
            "diagnosis": pred_cls,
            "ref_class": REF_CLASS,
            "pred_base": pred_cls,
            "prob": prob,
            "evidence_items": evidence_items,
            "model_profile": model_profile.get("key", ""),
            "model_name": model_profile.get("display_name", ""),
        })
        pred_rows.append({
            "patient_id": pid,
            "model_profile": model_profile.get("key", ""),
            "model_name": model_profile.get("display_name", ""),
            "pred_base": pred_cls,
            "prob_NC": prob.get("NC", 0.0),
            "prob_PRISm": prob.get("PRISm", 0.0),
            "prob_COPD": prob.get("COPD", 0.0),
        })

    return external_db, pd.DataFrame(pred_rows), missing


def strip_think(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"(?is)<\s*think\s*>.*?<\s*/\s*think\s*>", "", text).strip()


def post_clean(text: str) -> str:
    text = english_only_text(_TAG_RE.sub("", text or ""))
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def ensure_md_spacing(md: str) -> str:
    md = (md or "").replace("\r\n", "\n")
    md = re.sub(r"\n(###\s*)", r"\n\n\1", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def format_evidence_block(query_evid: List[Dict[str, Any]]) -> str:
    lines = [english_only_text(item.get("text", "")) for item in query_evid if english_only_text(item.get("text", ""))]
    return "\n".join([f"{i + 1}. {line}" for i, line in enumerate(lines)]) or "(none)"


def format_retrieved_cases(cases: List[dict], include_true: bool = False, max_cases: int = 3) -> str:
    if not cases:
        return "(none)"
    lines = []
    for case in cases[:max_cases]:
        cid = str(case.get("patient_id", "")).strip()
        sim = safe_float(case.get("sim"), 0.0)
        true_part = f" | true={normalize_label(case.get('true_label'))}" if include_true else ""
        evidence = "; ".join([english_only_text(x) for x in (case.get("matched_evidence") or [])[:2] if english_only_text(x)])
        lines.append(f"- {cid} | sim={sim:.3f}{true_part} | evidence: {evidence or '(none)'}")
    return "\n".join(lines)


def call_llm(client: Any, messages: List[Dict[str, str]], temperature: float, model_name: str = MODEL_NAME) -> str:
    last_exc: Optional[Exception] = None
    for attempt in range(1, NETWORK_MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(model=model_name or MODEL_NAME, messages=messages, temperature=temperature)
            return strip_think(response.choices[0].message.content or "")
        except Exception as exc:
            last_exc = exc
            if attempt < NETWORK_MAX_RETRIES:
                time.sleep(NETWORK_RETRY_SECONDS)
    raise RuntimeError(f"LLM request failed: {last_exc}")


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    cleaned = strip_think(text).strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    match = _JSON_OBJ_RE.search(cleaned)
    if not match:
        return None
    try:
        obj = json.loads(match.group(1))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def llm_update_probabilities(client: Any, patient_id: str, pred_base: str, prob: Dict[str, Any], margin: float, risk_level: str, risk_reason: str, decision: str, rule_final_label: str, stats: Dict[str, Any], query_evid: List[Dict[str, Any]], support: List[dict], confuse: List[dict], model_name: str = MODEL_NAME) -> Dict[str, Any]:
    system_msg = (
        "You are a respiratory imaging probability calibration assistant. "
        "Use the base ML probabilities as prior and update only when evidence supports it. "
        "Return strict JSON only. Use English class names and English text only."
    )
    user_msg = f"""
Patient ID: {patient_id}
Base prediction: {pred_base}
Base probabilities: {json.dumps(normalize_prob_dict(prob), ensure_ascii=True)}
Base margin: {margin:.3f}
Risk level: {risk_level}
Risk reason: {risk_reason}
Rule decision: {decision}
Rule final label: {rule_final_label}
Directional stats: {json.dumps(stats, ensure_ascii=True)}

Current patient evidence:
{format_evidence_block(query_evid[:8])}

Correct-library supports:
{format_retrieved_cases(support, include_true=False, max_cases=3)}

Error-library confusions:
{format_retrieved_cases(confuse, include_true=True, max_cases=3)}

Return one JSON object:
{{"final_label":"NC|PRISm|COPD","confidence":0.0,"class_probs":{{"NC":0.0,"PRISm":0.0,"COPD":0.0}}}}
""".strip()
    text = call_llm(client, [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}], TEMP_PROB_UPDATE, model_name=model_name)
    obj = extract_first_json_object(text)
    if not obj:
        raise RuntimeError("LLM probability update did not return valid JSON.")
    prob_final = normalize_prob_dict(obj.get("class_probs", obj.get("prob", {})))
    if not prob_final:
        raise RuntimeError("LLM probability update returned empty probabilities.")
    final_label = argmax_label(prob_final)
    return {
        "prob_final": prob_final,
        "final_label": final_label,
        "confidence": max(prob_final.values()),
        "raw_response": text,
    }


def render_section_6_supports(support: List[dict]) -> str:
    lines = ["### 6) Similar Correct Cases"]
    if not support:
        lines.append("No similar correct support case was retrieved.")
        return "\n".join(lines)
    for case in support:
        cid = str(case.get("patient_id", "")).strip()
        sim = safe_float(case.get("sim"), 0.0)
        evidence = [post_clean(str(x).strip()) for x in case.get("matched_evidence", []) if str(x).strip()]
        lines.append(f"**{cid}** (sim={sim:.3f})")
        lines.append(f"The case supports the current diagnostic direction through matched quantitative evidence. Evidence: {evidence[0] if evidence else 'none'}")
    return "\n\n".join(lines)


def render_section_7_confusions(confuse: List[dict], margin: float, risk_level: str, risk_reason: str) -> str:
    lines = ["### 7) Confusion Risk and Error Cases"]
    lines.append(f"Model confidence margin={margin:.3f}. Risk level: {risk_level}. {risk_reason}")
    if not confuse:
        lines.append("No similar historical error case was retrieved.")
        return "\n\n".join(lines)
    for case in confuse:
        cid = str(case.get("patient_id", "")).strip()
        sim = safe_float(case.get("sim"), 0.0)
        true_label = normalize_label(case.get("true_label"))
        evidence = [post_clean(str(x).strip()) for x in case.get("matched_evidence", []) if str(x).strip()]
        lines.append(f"**{cid}** (sim={sim:.3f}, true={true_label})")
        lines.append(f"This error case shares part of the current evidence pattern and should be considered during review. Evidence: {evidence[0] if evidence else 'none'}")
    return "\n\n".join(lines)


def replace_sections_6_7(md: str, sec6: str, sec7: str) -> str:
    if not md:
        return "\n\n".join([sec6, sec7]).strip()
    text = md.replace("\r\n", "\n")
    p6 = re.compile(r"### 6\) Similar Correct Cases.*?(?=\n### 7\) Confusion Risk and Error Cases)", re.S)
    if p6.search(text):
        text = p6.sub(sec6 + "\n\n", text)
    else:
        text = insert_before_section_8(text, sec6)
    p7 = re.compile(r"### 7\) Confusion Risk and Error Cases.*?(?=\n### 8\) Summary and Clinical Recommendation)", re.S)
    if p7.search(text):
        text = p7.sub(sec7 + "\n\n", text)
    else:
        text = insert_before_section_8(text, sec7)
    return ensure_md_spacing(text)


def insert_before_section_8(md: str, section: str) -> str:
    marker = "\n### 8) Summary and Clinical Recommendation"
    text = md or ""
    idx = text.find(marker)
    if idx >= 0:
        return text[:idx].rstrip() + "\n\n" + section + "\n\n" + text[idx:].lstrip()
    return text.strip() + "\n\n" + section


def ensure_report_has_8_sections(md: str) -> str:
    sections = [
        ("### 1) Gate Decision", "Gate decision text was not returned by the report model."),
        ("### 2) Final Conclusion", "Final conclusion text was not returned by the report model."),
        ("### 3) Model Confidence", "Model confidence text was not returned by the report model."),
        ("### 4) Evidence Chain", "Evidence chain text was not returned by the report model."),
        ("### 5) LLM Triage", "LLM triage text was not returned by the report model."),
        ("### 6) Similar Correct Cases", "Similar correct cases were not returned by the report model."),
        ("### 7) Confusion Risk and Error Cases", "Confusion risk text was not returned by the report model."),
        ("### 8) Summary and Clinical Recommendation", "Summary and clinical recommendation text was not returned by the report model."),
    ]
    text = md or ""
    missing = []
    for heading, fallback in sections:
        if not re.search(rf"^{re.escape(heading)}\s*$", text, flags=re.M):
            missing.append(f"{heading}\n{fallback}")
    if missing:
        text = text.strip() + "\n\n" + "\n\n".join(missing)
    return ensure_md_spacing(text)


def fallback_report(pid: str, pred_base: str, final_label: str, decision: str, gate_on: bool, gate_reasons: List[str], prob: Dict[str, Any], margin: float, risk_level: str, risk_reason: str, query_evid: List[Dict[str, Any]]) -> str:
    ev1 = query_evid[0].get("text") if query_evid else "No evidence available."
    return ensure_md_spacing(f"""
### 1) Gate Decision
gate_triggered={gate_on}. Reasons: {"; ".join(gate_reasons) or "none"}. Evidence: {ev1}

### 2) Final Conclusion
Base prediction is **{pred_base}** and final label is **{final_label}** with decision={decision}. Evidence: {ev1}

### 3) Model Confidence
Probability distribution: {format_prob(prob)}. margin={margin:.3f}. Risk: {risk_level} ({risk_reason}). Evidence: {ev1}

### 4) Evidence Chain
{format_evidence_block(query_evid[:6])}

### 5) LLM Triage
This Markdown narrative was generated in local deterministic mode. LLM narrative generation was not used for this report. Evidence: {ev1}

### 6) Similar Correct Cases
Similar correct cases will be inserted by the deterministic retrieval renderer.

### 7) Confusion Risk and Error Cases
Confusion and counterexample cases will be inserted by the deterministic retrieval renderer.

### 8) Summary and Clinical Recommendation
The final label is **{final_label}**. This output is intended to support clinical review and should be interpreted with symptoms, pulmonary function, and other clinical information. Evidence: {ev1}
""".strip())


def build_report_prompt(pid: str, pred_base: str, final_label: str, decision: str, decision_reasons: List[str], decision_notes: str, prob: Dict[str, Any], margin: float, gate_on: bool, gate_reasons: List[str], risk_level: str, risk_reason: str, query_evid: List[Dict[str, Any]], support: List[dict], confuse: List[dict]) -> List[Dict[str, str]]:
    system_msg = (
        "You are a clinical auxiliary explanation assistant. Output a structured English Markdown report.\n"
        "Hard rules:\n"
        "1) pred_base and probabilities are explanatory inputs and must not be altered.\n"
        "2) final_label has already been determined by the system and must be used exactly.\n"
        "3) Use natural paragraphs and avoid long bullet lists.\n"
        "4) Every section must include at least one current-case evidence sentence when evidence is available.\n"
        "5) Do not output Chinese or any non-English prose.\n"
        "6) Do not output chain-of-thought; only write auditable reasons.\n"
        "7) Sections 6 and 7 will be replaced by deterministic program-side content, but you must still output all 8 headings."
    )
    user_msg = f"""
Patient ID: {pid}
pred_base: {pred_base}
final_label: {final_label}
decision: {decision}
decision_reasons: {"; ".join(decision_reasons)}
decision_notes: {decision_notes}
probabilities: {json.dumps(prob, ensure_ascii=True)}
margin: {margin:.3f}
gate_triggered: {gate_on}
gate_reasons: {"; ".join(gate_reasons)}
risk: {risk_level} - {risk_reason}

Evidence:
{format_evidence_block(query_evid)}

Correct supports:
{format_retrieved_cases(support, include_true=False, max_cases=5)}

Confusions:
{format_retrieved_cases(confuse, include_true=True, max_cases=5)}

Use these headings:
### 1) Gate Decision
### 2) Final Conclusion
### 3) Model Confidence
### 4) Evidence Chain
### 5) LLM Triage
### 6) Similar Correct Cases
### 7) Confusion Risk and Error Cases
### 8) Summary and Clinical Recommendation

Section writing requirements:
- Section 1 must contain 2-4 clinically useful sentences about whether gate review was triggered and why.
- Section 2 must contain 2-4 sentences explaining final_label and its relationship to pred_base: kept, changed, or marked for review.
- Section 3 must contain one paragraph explaining the three-class probability ranking and the meaning of the margin.
- Section 4 must contain 3-6 short paragraphs, each explaining one current-patient evidence item and ending with "Evidence: ...".
- Section 5 must contain 1-2 paragraphs explaining how LLM-assisted review should be interpreted under gate_triggered and decision.
- Section 6 must include this heading and one placeholder paragraph; the program will replace it with deterministic similar-case content.
- Section 7 must include this heading and one placeholder paragraph; the program will replace it with deterministic confusion-case content.
- Section 8 must contain 4-6 connected sentences: final conclusion, key evidence, clinical information to check such as symptoms and pulmonary function, next-step recommendation, and a disclaimer that this AI-assisted output does not replace physician diagnosis.
""".strip()
    return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]


def emit_progress(progress_callback: Optional[ProgressCallback], step_key: str, state: str) -> None:
    if progress_callback is not None:
        progress_callback(step_key, state)


def _required_resource_path(
    path: Optional[str | Path],
    env_var: str,
    label: str,
) -> Path:
    candidate = path or os.getenv(env_var)
    if not candidate:
        raise FileNotFoundError(
            f"Required {label} file was not provided. "
            f"Pass the path or set {env_var}."
        )
    resolved = Path(candidate).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Required {label} file was not provided: {resolved}"
        )
    return resolved


def process_cases(
    df_ext: pd.DataFrame,
    use_llm: bool = False,
    api_key: Optional[str] = None,
    limit: Optional[int] = None,
    api_base_url: Optional[str] = None,
    model_name: Optional[str] = None,
    progress_callback: Optional[ProgressCallback] = None,
    model_path: Optional[str | Path] = None,
    feature_scores_path: Optional[str | Path] = None,
    train_stats_path: Optional[str | Path] = None,
    correct_evidence_path: Optional[str | Path] = None,
    error_evidence_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    model_path = _required_resource_path(model_path, "PGRAD_MODEL_PATH", "model")
    feature_scores_path = _required_resource_path(
        feature_scores_path,
        "PGRAD_FEATURE_SCORES_PATH",
        "feature-score",
    )
    train_stats_path = _required_resource_path(
        train_stats_path,
        "PGRAD_TRAIN_STATS_PATH",
        "reference-statistics",
    )
    correct_evidence_path = _required_resource_path(
        correct_evidence_path,
        "PGRAD_CORRECT_EVIDENCE_PATH",
        "correct-evidence",
    )
    error_evidence_path = _required_resource_path(
        error_evidence_path,
        "PGRAD_ERROR_EVIDENCE_PATH",
        "error-evidence",
    )

    emit_progress(progress_callback, "match_features", "running")
    df_ext = df_ext.copy()
    df_ext.columns = [str(col).strip() for col in df_ext.columns]
    model_profile, bundle, _ = select_model_profile_for_upload(df_ext, model_path)
    emit_progress(progress_callback, "match_features", "complete")

    emit_progress(progress_callback, "load_assets", "running")
    ref_stats = load_ref_stats(train_stats_path)
    correct_db = load_jsonl(correct_evidence_path)
    error_db = load_jsonl(error_evidence_path)
    emit_progress(progress_callback, "load_assets", "complete")

    emit_progress(progress_callback, "build_evidence", "running")
    external_db, base_predictions, missing = build_external_evidence(
        df_ext,
        bundle,
        ref_stats,
        model_profile,
        feature_scores_path,
    )
    if limit is not None and limit > 0:
        external_db = external_db[:limit]
    emit_progress(progress_callback, "build_evidence", "complete")

    emit_progress(progress_callback, "setup_llm", "running")
    client = None
    selected_model = model_name or MODEL_NAME
    if use_llm:
        key = api_key or os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("LLM is enabled but no API key was provided.")
        if OpenAI is None:
            raise RuntimeError("openai package is not installed.")
        selected_base_url = api_base_url or os.getenv("QWEN_BASE_URL") or os.getenv("OPENAI_BASE_URL") or BASE_URL
        client = OpenAI(api_key=key, base_url=selected_base_url)
    emit_progress(progress_callback, "setup_llm", "complete")

    emit_progress(progress_callback, "gate_reports", "running")
    cases = []
    for obj in external_db:
        pid = str(obj.get("patient_id", "")).strip()
        pred_base = normalize_label(obj.get("pred_base", ""))
        prob = normalize_prob_dict(obj.get("prob", {}))
        margin = top2_margin(prob)
        query_evid = topk_evidence(parse_evidence_items(obj), QUERY_TOPK_EVID)
        support, confuse = retrieve_cases(pred_base, query_evid, correct_db, error_db)
        risk_level, risk_reason = precompute_risk(prob, confuse)
        max_sim = max([safe_float(item.get("sim"), 0.0) for item in confuse], default=0.0)
        gate_on, gate_reasons = gate_trigger(margin, max_sim, risk_level)
        decision, rule_final_label, review_flag, decision_reasons, decision_notes, stats = directional_decision(pred_base, prob, margin, gate_on, support, confuse)

        prob_final = dict(prob)
        final_label = rule_final_label
        confidence = max(prob_final.values()) if prob_final else 0.0
        llm_used = False
        llm_error = ""
        decision_key = str(decision or "").strip().lower()
        allow_llm_probability_update = gate_on or margin <= MARGIN_WEAK or decision_key in {"switch", "abstain", "review"}
        if use_llm and client is not None and allow_llm_probability_update:
            try:
                llm_result = llm_update_probabilities(client, pid, pred_base, prob, margin, risk_level, risk_reason, decision, rule_final_label, stats, query_evid, support, confuse, model_name=selected_model)
                prob_final = llm_result["prob_final"]
                final_label = llm_result["final_label"]
                confidence = llm_result["confidence"]
                llm_used = True
            except Exception as exc:
                llm_error = str(exc)

        margin_final = top2_margin(prob_final)
        md = fallback_report(pid, pred_base, final_label, decision, gate_on, gate_reasons, prob_final, margin_final, risk_level, risk_reason, query_evid)
        if use_llm and client is not None:
            try:
                messages = build_report_prompt(pid, pred_base, final_label, decision, decision_reasons, decision_notes, prob_final, margin_final, gate_on, gate_reasons, risk_level, risk_reason, query_evid, support, confuse)
                md = ensure_md_spacing(post_clean(call_llm(client, messages, TEMP_REPORT, model_name=selected_model)))
            except Exception as exc:
                llm_error = llm_error or str(exc)
        sec6 = render_section_6_supports(support)
        sec7 = render_section_7_confusions(confuse, margin_final, risk_level, risk_reason)
        md = replace_sections_6_7(md, sec6, sec7)
        md = ensure_report_has_8_sections(md)

        row = {
            "patient_id": pid,
            "model_profile": obj.get("model_profile", model_profile.get("key", "")),
            "model_name": obj.get("model_name", model_profile.get("display_name", "")),
            "pred_base": pred_base,
            "rule_final_label": rule_final_label,
            "pred_label": final_label,
            "confidence": confidence,
            "p_NC": prob_final.get("NC", 0.0),
            "p_PRISm": prob_final.get("PRISm", 0.0),
            "p_COPD": prob_final.get("COPD", 0.0),
            "p_base_NC": prob.get("NC", 0.0),
            "p_base_PRISm": prob.get("PRISm", 0.0),
            "p_base_COPD": prob.get("COPD", 0.0),
            "decision": decision,
            "gate_triggered": bool(gate_on),
            "review_flag": bool(review_flag),
            "risk_level": risk_level,
            "risk_reason": risk_reason,
            "margin_base": margin,
            "margin_final": margin_final,
            "S_correct_max": stats.get("S_correct_max", 0.0),
            "S_confuse_max": stats.get("S_confuse_max", 0.0),
            "delta_sim": stats.get("delta_sim", 0.0),
            "llm_used": llm_used,
            "llm_error": llm_error,
            "llm_model": selected_model if use_llm else "",
        }
        cases.append({
            **row,
            "markdown": md,
            "query_evidence": query_evid,
            "support": support,
            "confuse": confuse,
            "external_object": obj,
        })
    emit_progress(progress_callback, "gate_reports", "complete")

    emit_progress(progress_callback, "finalize", "running")
    result_df = pd.DataFrame([{key: value for key, value in case.items() if key not in ["markdown", "query_evidence", "support", "confuse", "external_object"]} for case in cases])
    result = {
        "cases": cases,
        "results": result_df,
        "base_predictions": base_predictions,
        "missing": missing,
        "model_profile": model_profile,
    }
    emit_progress(progress_callback, "finalize", "complete")
    return result
