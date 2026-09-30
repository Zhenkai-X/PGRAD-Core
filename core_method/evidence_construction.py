# -*- coding: utf-8 -*-
"""Evidence construction for PGRAD.

Evidence is constructed from predictors that survived preprocessing and are
actually available in the current case. Missing expiratory, subtraction, or
ParaMap variables are not imputed as evidence for single-inspiratory cases.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


VALID_CLASSES = ["NC", "PRISm", "COPD"]
REF_CLASS = "NC"
TOP_K_EVIDENCE = 30
Z_NEAR = 0.25
ID_COL_CANDIDATES = [
    "ID",
    "patient_id",
    "PatientID",
    "subject_id",
    "RID",
    "PTID",
    "id",
]


@dataclass
class RefStats:
    ref: Dict[str, Dict[str, object]] = field(default_factory=dict)
    mean_by_class: Dict[str, Dict[str, float]] = field(default_factory=dict)


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, engine="openpyxl")
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path, engine="python")


def normalize_id(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
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
    if low in {"nc", "normal", "nl", "healthy", "control", "0"}:
        return "NC"
    if low in {"prism", "pris", "pr", "1"}:
        return "PRISm"
    if low in {"copd", "2"}:
        return "COPD"
    return text


def detect_id_col(df: pd.DataFrame, preferred: Optional[str] = None) -> str:
    if preferred and preferred in df.columns:
        return preferred
    for col in ID_COL_CANDIDATES:
        if col in df.columns:
            return col
    for col in df.columns:
        if "id" in str(col).lower():
            return str(col)
    raise ValueError("No patient ID column was found.")


def fmt_value(value: float) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    absolute = abs(float(value))
    if absolute >= 1000 or (0 < absolute < 1e-3):
        return f"{float(value):.6g}"
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def load_predictors(path: str | Path) -> List[str]:
    df = read_table(path)
    if "selected_features" in df.columns:
        return [str(x).strip() for x in df["selected_features"].dropna()]
    if "feature" in df.columns:
        return [str(x).strip() for x in df["feature"].dropna()]
    first = df.columns[0]
    return [str(x).strip() for x in df[first].dropna()]


def load_fused_scores(path: str | Path, predictors: Iterable[str]) -> Dict[str, float]:
    predictors = list(predictors)
    scores: Dict[str, float] = {}
    df = read_table(path)
    if "feature" not in df.columns or "fused_score" not in df.columns:
        raise ValueError("Feature score table must contain feature and fused_score columns.")
    for _, row in df.iterrows():
        feat = str(row["feature"]).strip()
        if feat in predictors:
            try:
                scores[feat] = float(row["fused_score"])
            except Exception:
                pass
    missing = [feat for feat in predictors if feat not in scores]
    if missing:
        raise ValueError(f"Missing fused_score for predictors: {missing[:10]}")
    return scores


def empirical_percentile(sorted_arr: np.ndarray, value: float) -> int:
    if sorted_arr is None or len(sorted_arr) == 0:
        return 50
    idx = np.searchsorted(sorted_arr, value, side="right")
    return max(0, min(100, int(round(100.0 * idx / len(sorted_arr)))))


def build_reference_stats(
    train_df: pd.DataFrame,
    predictors: List[str],
    label_col: str = "label",
    ref_class: str = REF_CLASS,
) -> RefStats:
    stats = RefStats()
    if label_col not in train_df.columns:
        raise ValueError(f"Label column not found in reference data: {label_col}")

    tmp = train_df.copy()
    tmp[label_col] = tmp[label_col].apply(normalize_label)
    tmp = tmp[tmp[label_col].isin(VALID_CLASSES)].reset_index(drop=True)

    for feat in predictors:
        if feat not in tmp.columns:
            continue
        stats.mean_by_class[feat] = {}
        for cls in VALID_CLASSES:
            values = pd.to_numeric(
                tmp.loc[tmp[label_col] == cls, feat], errors="coerce"
            ).dropna()
            if len(values):
                stats.mean_by_class[feat][cls] = float(values.mean())

    ref = tmp[tmp[label_col] == ref_class]
    for feat in predictors:
        if feat not in ref.columns:
            continue
        values = pd.to_numeric(ref[feat], errors="coerce").dropna()
        if len(values) < 3:
            continue
        arr = values.to_numpy(dtype=float)
        stats.ref[feat] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr) + 1e-12),
            "sorted": np.sort(arr),
        }
    return stats


def z_and_percentile(
    ref_stats: RefStats, feat: str, value: float
) -> Tuple[Optional[float], Optional[int]]:
    if feat not in ref_stats.ref:
        return None, None
    ref = ref_stats.ref[feat]
    z_value = (float(value) - float(ref["mean"])) / (float(ref["std"]) + 1e-12)
    pct = empirical_percentile(ref["sorted"], float(value))
    return float(z_value), int(pct)


def direction_text(z_value: float) -> str:
    if abs(z_value) < Z_NEAR:
        return "near"
    return "high" if z_value > 0 else "low"


def support_direction(
    ref_stats: RefStats, feat: str, z_value: float, pred_cls: str
) -> str:
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


def construct_case_evidence(
    row: pd.Series,
    patient_id: str,
    pred_base: str,
    prob: Dict[str, float],
    predictors: List[str],
    ref_stats: RefStats,
    fused_scores: Dict[str, float],
    top_k: int = TOP_K_EVIDENCE,
) -> Dict[str, Any]:
    candidates = []
    for feat in predictors:
        if feat not in row.index:
            continue
        value = pd.to_numeric(pd.Series([row.get(feat)]), errors="coerce").iloc[0]
        if pd.isna(value):
            continue
        z_value, pct = z_and_percentile(ref_stats, feat, float(value))
        if z_value is None:
            continue
        strength = abs(float(z_value)) * float(fused_scores.get(feat, 1.0))
        candidates.append((strength, feat, float(value), float(z_value), int(pct)))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[:top_k]

    evidence_items = []
    knowledge_graph = []
    for strength, feat, value, z_value, pct in selected:
        value_str = fmt_value(value)
        text = (
            f"{feat}={value_str}; relative to {REF_CLASS}: {direction_text(z_value)}; "
            f"z={z_value:+0.2f}; approx P{pct}"
        )
        direction = support_direction(ref_stats, feat, z_value, pred_base)
        if direction == "consistent":
            text += f"; consistent with {pred_base}"
        elif direction == "opposite":
            text += f"; opposite to {pred_base}"
        item = {
            "feature": feat,
            "value": value,
            "value_str": value_str,
            "z": z_value,
            "percentile": pct,
            "fused_score": float(fused_scores.get(feat, 1.0)),
            "score": float(strength),
            "text": text,
        }
        evidence_items.append(item)
        knowledge_graph.append([value_str, text, feat])

    knowledge_graph.append([patient_id, f"Prior prediction is {pred_base}", pred_base])
    return {
        "patient_id": patient_id,
        "pred_base": pred_base,
        "diagnosis": pred_base,
        "prob": prob,
        "ref_class": REF_CLASS,
        "knowledge_graph": knowledge_graph,
        "evidence_items": evidence_items,
    }


def build_evidence_table(
    current_df: pd.DataFrame,
    prior_df: pd.DataFrame,
    train_df: pd.DataFrame,
    predictors: List[str],
    fused_scores: Dict[str, float],
    id_col: Optional[str] = None,
    label_col: str = "label",
) -> List[Dict[str, Any]]:
    id_col = detect_id_col(current_df, id_col)
    current = current_df.copy()
    current[id_col] = current[id_col].apply(normalize_id)
    prior = prior_df.copy()
    prior_id_col = detect_id_col(prior)
    prior[prior_id_col] = prior[prior_id_col].apply(normalize_id)
    prior_map = prior.set_index(prior_id_col).to_dict(orient="index")
    ref_stats = build_reference_stats(train_df, predictors, label_col=label_col)

    outputs = []
    for _, row in current.iterrows():
        pid = normalize_id(row[id_col])
        if pid not in prior_map:
            continue
        prior_row = prior_map[pid]
        prob = {
            cls: float(prior_row.get(f"prob_{cls}", 0.0))
            for cls in VALID_CLASSES
        }
        pred_base = normalize_label(prior_row.get("pred_base"))
        if pred_base not in VALID_CLASSES:
            pred_base = max(prob, key=prob.get)
        outputs.append(
            construct_case_evidence(
                row,
                pid,
                pred_base,
                prob,
                predictors,
                ref_stats,
                fused_scores,
            )
        )
    return outputs


def write_jsonl(rows: List[Dict[str, Any]], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct PGRAD evidence.")
    parser.add_argument("--current", required=True, help="Current-case CSV/XLSX.")
    parser.add_argument(
        "--prior",
        required=True,
        help="Prior predictions with prob_NC/prob_PRISm/prob_COPD.",
    )
    parser.add_argument("--train", required=True, help="Training/reference CSV/XLSX.")
    parser.add_argument("--predictors", required=True, help="Retained predictor list.")
    parser.add_argument("--feature-scores", required=True, help="Scores with fused_score.")
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--label-col", default="label")
    args = parser.parse_args()

    predictors = load_predictors(args.predictors)
    fused_scores = load_fused_scores(args.feature_scores, predictors)
    rows = build_evidence_table(
        read_table(args.current),
        read_table(args.prior),
        read_table(args.train),
        predictors,
        fused_scores,
        label_col=args.label_col,
    )
    write_jsonl(rows, args.out_jsonl)
    print(json.dumps({"n_cases": len(rows), "out_jsonl": args.out_jsonl}, indent=2))


if __name__ == "__main__":
    main()
