# -*- coding: utf-8 -*-
"""Public implementation of the formal PGRAD prior-model method."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from .config_loader import load_config


CONFIG = load_config()
PROJECT = CONFIG["project"]
PRIOR = CONFIG["prior_model"]
VALID_CLASSES = list(PROJECT["classes"])
SEED = int(PROJECT["random_seed"])
TOP_K_PREDICTORS = int(PRIOR["top_k_predictors"])
N_SPLITS = int(PRIOR["n_splits"])
MISSING_THRESHOLD = float(PRIOR["missing_threshold"])
CORRELATION_THRESHOLD = float(PRIOR["correlation_threshold"])
NEAR_CONSTANT_THRESHOLD = float(PRIOR["near_constant_threshold"])
MAJOR_CLASS = str(PRIOR["class_balance"]["major_class"])
MAJOR_CLASS_FACTOR = int(PRIOR["class_balance"]["major_class_factor"])


def read_table(path: str | Path) -> pd.DataFrame:
    """Read CSV or Excel input."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, engine="openpyxl")
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception:
            continue
    return pd.read_csv(path, engine="python")


def normalize_label(value: Any) -> str:
    """Normalize common class encodings."""
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


def numeric_predictors(df: pd.DataFrame, exclude: Iterable[str]) -> List[str]:
    """Return numeric predictors excluding identifier and label columns."""
    blocked = set(exclude)
    return [
        str(column)
        for column in df.columns
        if column not in blocked and pd.api.types.is_numeric_dtype(df[column])
    ]


def corr_prune(
    df: pd.DataFrame,
    predictors: List[str],
    correlation_threshold: float = CORRELATION_THRESHOLD,
) -> List[str]:
    """Drop one feature from each pair with absolute correlation above threshold."""
    if len(predictors) <= 1:
        return list(predictors)
    correlation = df[predictors].astype(float).corr().abs()
    upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
    to_drop = set()
    for column in upper.columns:
        if column in to_drop:
            continue
        to_drop.update(
            upper.index[upper[column] > correlation_threshold].tolist()
        )
    return [column for column in predictors if column not in to_drop]


def preprocess_predictors(
    df: pd.DataFrame,
    predictors: List[str],
    missing_threshold: float = MISSING_THRESHOLD,
    near_constant_ratio: float = NEAR_CONSTANT_THRESHOLD,
) -> List[str]:
    """Apply the formal missingness, near-constant and correlation filters."""
    keep = []
    for column in predictors:
        series = pd.to_numeric(df[column], errors="coerce")
        if series.isna().mean() > missing_threshold:
            continue
        if series.nunique(dropna=True) <= 1:
            continue
        top_frequency = series.value_counts(dropna=False, normalize=True).max()
        if series.nunique(dropna=False) <= 3 and top_frequency > near_constant_ratio:
            continue
        keep.append(column)
    if not keep:
        return []
    imputed = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(df[keep]),
        columns=keep,
        index=df.index,
    )
    return corr_prune(imputed, keep)


def normalize_score(scores: pd.Series) -> pd.Series:
    """Min-max normalize one feature-score source."""
    scores = scores.astype(float)
    if scores.max() - scores.min() < 1e-12:
        return scores * 0.0
    return (scores - scores.min()) / (scores.max() - scores.min())


def mrmr_like(
    df: pd.DataFrame,
    predictors: List[str],
    y: np.ndarray,
    k: int = 40,
    random_state: int = SEED,
) -> Tuple[List[str], dict[str, float]]:
    """Calculate MI and an mRMR-like nonredundant ordering."""
    values = df[predictors].astype(float).to_numpy()
    mutual_information = mutual_info_classif(
        values, y, discrete_features=False, random_state=random_state
    )
    mi_map = {
        predictors[index]: float(mutual_information[index])
        for index in range(len(predictors))
    }
    correlation = df[predictors].astype(float).corr().abs()
    selected = []
    remaining = set(predictors)
    while len(selected) < min(k, len(predictors)):
        best_feature, best_score = None, -1e18
        for feature in remaining:
            redundancy = correlation.loc[feature, selected].mean() if selected else 0.0
            score = mi_map.get(feature, 0.0) - redundancy
            if score > best_score:
                best_feature, best_score = feature, score
        selected.append(best_feature)
        remaining.remove(best_feature)
    return selected, mi_map


def controlled_downsample_indices(
    y_sub: np.ndarray,
    class_names: List[str],
    major_class: str = MAJOR_CLASS,
    factor: int = MAJOR_CLASS_FACTOR,
    random_state: np.random.RandomState | None = None,
) -> np.ndarray:
    """Apply formal controlled downsampling to the major class."""
    random_state = random_state or np.random.RandomState(SEED)
    if major_class not in class_names:
        return np.arange(len(y_sub))
    major_code = class_names.index(major_class)
    all_indices = np.arange(len(y_sub))
    major_indices = all_indices[y_sub == major_code]
    minor_indices = all_indices[y_sub != major_code]
    if not len(major_indices) or not len(minor_indices):
        return all_indices
    keep_count = min(len(major_indices), factor * len(minor_indices))
    selected_major = (
        random_state.choice(major_indices, size=keep_count, replace=False)
        if keep_count < len(major_indices)
        else major_indices
    )
    selected = np.concatenate([selected_major, minor_indices])
    random_state.shuffle(selected)
    return selected


def make_l1_logit(seed: int = SEED) -> Pipeline:
    """Create the formal median-imputed, standardized L1-logistic pipeline."""
    params = dict(PRIOR["logistic_regression"])
    params.update({"random_state": seed, "multi_class": "multinomial"})
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(**params)),
        ]
    )


def cv_lgbm_importance(
    df: pd.DataFrame,
    predictors: List[str],
    y: np.ndarray,
    class_names: List[str],
    n_splits: int = N_SPLITS,
    seed: int = SEED,
) -> pd.Series:
    """Average LightGBM feature importance across CV folds."""
    params = dict(PRIOR["gradient_boosting"])
    params.update({"random_state": seed, "n_jobs": -1})
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    random_state = check_random_state(seed)
    importance = pd.Series(0.0, index=predictors)
    for train_index, _ in splitter.split(df[predictors], y):
        local_index = controlled_downsample_indices(
            y[train_index], class_names, random_state=random_state
        )
        train_use = train_index[local_index]
        model = LGBMClassifier(**params)
        model.fit(df.iloc[train_use][predictors], y[train_use])
        importance += pd.Series(model.feature_importances_, index=predictors)
    return importance / n_splits


def cv_logit_scores(
    df: pd.DataFrame,
    predictors: List[str],
    y: np.ndarray,
    class_names: List[str],
    n_splits: int = N_SPLITS,
    seed: int = SEED,
) -> Tuple[pd.Series, pd.Series]:
    """Average L1 coefficient magnitude and nonzero selection stability."""
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    random_state = check_random_state(seed)
    coefficient_sum = pd.Series(0.0, index=predictors)
    nonzero_count = pd.Series(0.0, index=predictors)
    for train_index, _ in splitter.split(df[predictors], y):
        local_index = controlled_downsample_indices(
            y[train_index], class_names, random_state=random_state
        )
        train_use = train_index[local_index]
        pipeline = make_l1_logit(seed)
        pipeline.fit(df.iloc[train_use][predictors], y[train_use])
        coefficient_abs = np.mean(
            np.abs(pipeline.named_steps["classifier"].coef_), axis=0
        )
        coefficient_sum += pd.Series(coefficient_abs, index=predictors)
        nonzero_count += (coefficient_abs > 1e-8).astype(float)
    return coefficient_sum / n_splits, nonzero_count / n_splits


def compute_feature_scores(
    df: pd.DataFrame,
    predictors: List[str],
    y: np.ndarray,
    class_names: List[str] | None = None,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Fuse MI, LightGBM, L1 magnitude and stability scores."""
    class_names = class_names or VALID_CLASSES
    _, mi_map = mrmr_like(
        df, predictors, y, k=min(40, len(predictors)), random_state=seed
    )
    model_importance = cv_lgbm_importance(
        df, predictors, y, class_names, n_splits, seed
    )
    coefficient_magnitude, selection_stability = cv_logit_scores(
        df, predictors, y, class_names, n_splits, seed
    )
    features = sorted(predictors)
    mi = pd.Series(mi_map).reindex(features).fillna(0.0)
    model = model_importance.reindex(features).fillna(0.0)
    coefficient = coefficient_magnitude.reindex(features).fillna(0.0)
    stability = selection_stability.reindex(features).fillna(0.0)
    weights = PRIOR["feature_score"]
    fused = (
        float(weights["mutual_information"]) * normalize_score(mi)
        + float(weights["model_importance"]) * normalize_score(model)
        + float(weights["coefficient_magnitude"]) * normalize_score(coefficient)
        + float(weights["selection_stability"]) * stability
    )
    return pd.DataFrame(
        {
            "feature": features,
            "mi": mi.values,
            "lgb_importance": model.values,
            "logit_coef_abs": coefficient.values,
            "logit_nz_frac": stability.values,
            "fused_score": fused.reindex(features).values,
        }
    ).sort_values("fused_score", ascending=False).reset_index(drop=True)


def cross_validated_prior(
    df: pd.DataFrame,
    predictors: List[str],
    y: np.ndarray,
    class_names: List[str],
    n_splits: int = N_SPLITS,
    seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate formal OOF prior predictions and probabilities."""
    oof_probability = np.zeros((len(df), len(class_names)), dtype=float)
    oof_prediction = np.zeros(len(df), dtype=int)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    random_state = check_random_state(seed)
    for train_index, valid_index in splitter.split(df[predictors], y):
        local_index = controlled_downsample_indices(
            y[train_index], class_names, random_state=random_state
        )
        train_use = train_index[local_index]
        pipeline = make_l1_logit(seed)
        pipeline.fit(df.iloc[train_use][predictors], y[train_use])
        probability = pipeline.predict_proba(df.iloc[valid_index][predictors])
        oof_probability[valid_index] = probability
        oof_prediction[valid_index] = np.argmax(probability, axis=1)
    return oof_prediction, oof_probability


def train_prior_model(
    input_path: str | Path,
    output_dir: str | Path,
    id_col: str = "ID",
    label_col: str = "label",
    top_k: int = TOP_K_PREDICTORS,
) -> dict:
    """Train and export the public prior-model artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = read_table(input_path)
    if label_col not in df.columns:
        raise ValueError(f"Label column not found: {label_col}")
    if id_col not in df.columns:
        df[id_col] = [f"case_{index + 1}" for index in range(len(df))]
    df[label_col] = df[label_col].apply(normalize_label)
    df = df[df[label_col].isin(VALID_CLASSES)].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No valid NC/PRISm/COPD cases remain after label normalization.")
    y = pd.Categorical(df[label_col], categories=VALID_CLASSES).codes
    predictors = preprocess_predictors(
        df, numeric_predictors(df, [id_col, label_col])
    )
    if len(predictors) < top_k:
        raise RuntimeError(
            f"Only {len(predictors)} usable predictors found; Top {top_k} requested."
        )
    feature_scores = compute_feature_scores(df, predictors, y, VALID_CLASSES)
    selected = feature_scores.head(top_k)["feature"].tolist()
    oof_prediction, oof_probability = cross_validated_prior(
        df, selected, y, VALID_CLASSES
    )
    final_model = make_l1_logit(SEED)
    final_index = controlled_downsample_indices(y, VALID_CLASSES)
    final_model.fit(df.iloc[final_index][selected], y[final_index])

    prediction_df = pd.DataFrame(
        {id_col: df[id_col].astype(str), "true_label": df[label_col]}
    )
    prediction_df["pred_base"] = [VALID_CLASSES[index] for index in oof_prediction]
    for index, class_name in enumerate(VALID_CLASSES):
        prediction_df[f"prob_{class_name}"] = oof_probability[:, index]
    prediction_df.to_csv(
        output_dir / "oof_prior_probabilities.csv",
        index=False,
        encoding="utf-8-sig",
    )
    feature_scores.to_csv(
        output_dir / "feature_scores.csv", index=False, encoding="utf-8-sig"
    )
    pd.Series(selected, name="selected_features").to_csv(
        output_dir / "selected_features.csv",
        index=False,
        encoding="utf-8-sig",
    )
    report = classification_report(
        y, oof_prediction, target_names=VALID_CLASSES, output_dict=True
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    joblib.dump(
        {
            "pipeline": final_model,
            "selected_features": selected,
            "class_names": VALID_CLASSES,
            "id_col": id_col,
            "label_col": label_col,
        },
        output_dir / "final_model_bundle.joblib",
        compress=3,
    )
    return {
        "selected_features": selected,
        "feature_scores": str(output_dir / "feature_scores.csv"),
        "oof_predictions": str(output_dir / "oof_prior_probabilities.csv"),
        "model_bundle": str(output_dir / "final_model_bundle.joblib"),
    }


def main() -> None:
    """Run prior-model training from the command line."""
    parser = argparse.ArgumentParser(description="Train the PGRAD Top-20 prior model.")
    parser.add_argument("--input", required=True, help="Training CSV/XLSX with label column.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--id-col", default="ID")
    parser.add_argument("--label-col", default="label")
    args = parser.parse_args()
    print(
        json.dumps(
            train_prior_model(args.input, args.out_dir, args.id_col, args.label_col),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
