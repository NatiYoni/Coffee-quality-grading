"""Offline, group-held-out evaluation of the existing cupping-score recipe."""

import argparse
import csv
import hashlib
import io
import json
import platform
import re
import sys
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

TARGET = "Total Cup Points"
SENSORY_FEATURES = (
    "Aroma", "Flavor", "Aftertaste", "Acidity", "Body", "Balance",
    "Uniformity", "Clean Cup", "Sweetness",
)
# Explicitly match the served manifest, not the notebook's dtype-based selection.
FEATURES = (
    "Altitude", *SENSORY_FEATURES, "Moisture Percentage",
    "Category One Defects", "Category Two Defects",
)
GROUP_COLUMNS = ("Country of Origin", "Farm Name")
SOURCE_REFERENCE = "https://www.kaggle.com/datasets/fatihb/coffee-quality-data-cqi"
UNKNOWN_NAMES = {
    "", "-", "?", "n/a", "na", "nan", "none", "null", "unknown",
    "not specified", "not available", "no data",
}
LIMITATIONS = [
    "The input hash identifies bytes, not provenance. The notebook requests an "
    "unpinned Kaggle dataset; the evaluator does not authenticate the supplied CSV.",
    "Nine inputs are cupping-score components. Row/group isolation does not remove "
    "this target relationship or establish forecasting before cupping.",
    "This is a fresh fit of a fixed legacy recipe, not validation of the saved or "
    "deployed artifact. Its hyperparameters were previously selected in the notebook.",
    "Country/farm matching is a name heuristic, not verified entity resolution. "
    "Unknown farms and aliases can hide related lots; identical names can over-group.",
    "Only exact parsed input profiles are linked as duplicates, not near-duplicates.",
    "This single split is not temporal, country-held-out, or an external benchmark. "
    "All input countries are included; there is no Ethiopia-only filter.",
    "The component-sum diagnostic uses Overall, an extra score component unavailable "
    "to the model. It is unadjusted for defect penalties and is not a fair predictor baseline.",
]


def parse_altitude(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", text)
    if match:
        low, high = map(float, match.groups())
        if low > high:
            raise ValueError("Altitude range must be ordered low-high.")
        result = (low + high) / 2
    else:
        try:
            result = float(text)
        except ValueError as error:
            raise ValueError(
                "Altitude must be numeric metres, a low-high range, or missing."
            ) from error
    if not np.isfinite(result) or result < 0:
        raise ValueError("Altitude must be finite and non-negative when present.")
    return result


def numeric_column(values: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    if (values.notna() & numeric.isna()).any():
        raise ValueError(f"{name} contains non-numeric values.")
    if np.isinf(numeric).any():
        raise ValueError(f"{name} contains infinite values.")
    if numeric.lt(0).any():
        raise ValueError(f"{name} contains negative values.")
    if name in (*SENSORY_FEATURES, "Overall") and numeric.gt(10).any():
        raise ValueError(f"{name} must be between 0 and 10 when present.")
    if name in (TARGET, "Moisture Percentage") and numeric.gt(100).any():
        raise ValueError(f"{name} must be between 0 and 100 when present.")
    if name in ("Category One Defects", "Category Two Defects"):
        if (numeric.dropna() % 1 != 0).any():
            raise ValueError(f"{name} must contain whole counts when present.")
    return numeric


def load_data(path: Path) -> tuple[pd.DataFrame, dict]:
    content = path.read_bytes()
    text = content.decode("utf-8-sig")
    records = csv.reader(io.StringIO(text), strict=True)
    header = next(records, [])
    if not header:
        raise ValueError("CSV is empty or has no header.")
    if len(header) != len(set(header)):
        raise ValueError("CSV has duplicate column names.")
    for row_number, row in enumerate(records, start=2):
        if row and len(row) != len(header):
            raise ValueError(f"CSV record {row_number} has the wrong number of fields.")
    data = pd.read_csv(io.StringIO(text))
    missing = sorted(set((*FEATURES, TARGET, *GROUP_COLUMNS)) - set(data.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
    if data.empty:
        raise ValueError("CSV has no data rows.")
    raw_duplicate_rows = int(data.duplicated().sum())
    data["Altitude"] = data["Altitude"].map(parse_altitude)
    for name in (*FEATURES[1:], TARGET, "Overall"):
        if name in data:
            data[name] = numeric_column(data[name], name)
    if data[TARGET].isna().any():
        raise ValueError("Total Cup Points has missing targets; no rows are silently dropped.")
    return data, {
        "path": str(path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "rows": len(data),
        "raw_duplicate_rows": raw_duplicate_rows,
        "raw_duplicate_definition": "Repeated full pandas-parsed rows, before feature normalization",
        "provenance_status": "unverified_user_supplied_local_csv",
        "expected_source_reference": SOURCE_REFERENCE,
        "source_version": None,
        "source_notebook": "notebooks/coffee_quality_model.ipynb",
        "excluded_from_model": sorted(set(data.columns) - set(FEATURES) - {TARGET}),
        "missing_features": {name: int(data[name].isna().sum()) for name in FEATURES},
    }


def normalized_name(value: object) -> str:
    if pd.isna(value):
        return ""
    name = " ".join(str(value).casefold().split())
    return "" if name in UNKNOWN_NAMES else name


def build_groups(data: pd.DataFrame) -> tuple[np.ndarray, dict]:
    parents = list(range(len(data)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def link(left: int, right: int) -> None:
        left, right = root(left), root(right)
        parents[max(left, right)] = min(left, right)

    farms: dict[tuple, int] = {}
    profiles: dict[tuple, int] = {}
    profile_targets: dict[tuple, set] = {}
    unknown_farm_rows = 0
    names = data.loc[:, GROUP_COLUMNS].itertuples(index=False, name=None)
    values = data.loc[:, FEATURES].itertuples(index=False, name=None)
    for index, (farm_values, feature_values) in enumerate(zip(names, values)):
        farm = tuple(normalized_name(value) for value in farm_values)
        if all(farm):
            link(index, farms.setdefault(farm, index))
        else:
            unknown_farm_rows += 1
        # Missingness is part of the profile; no fitted imputation is used for splitting.
        profile = tuple(None if pd.isna(value) else float(value) for value in feature_values)
        link(index, profiles.setdefault(profile, index))
        profile_targets.setdefault(profile, set()).add(float(data[TARGET].iloc[index]))
    groups = np.array([root(index) for index in range(len(data))])
    counts = Counter(groups)
    return groups, {
        "group_columns": list(GROUP_COLUMNS),
        "named_farm_groups": len(farms),
        "unknown_farm_rows": unknown_farm_rows,
        "duplicate_feature_rows": len(data) - len(profiles),
        "feature_profiles_with_conflicting_targets": sum(
            len(targets) > 1 for targets in profile_targets.values()
        ),
        "connected_groups": len(counts),
        "largest_group_rows": max(counts.values()),
    }


def split_data(
    data: pd.DataFrame, groups: np.ndarray, seed: int, test_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer between 0 and 2**32 - 1.")
    if not 0 < test_size < 1:
        raise ValueError("Test size must be strictly between 0 and 1.")
    if len(np.unique(groups)) < 2:
        raise ValueError("At least two independent groups are required.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train, test = next(splitter.split(data, groups=groups))
    if np.intersect1d(train, test).size or np.intersect1d(groups[train], groups[test]).size:
        raise ValueError("Split isolation failed: overlapping rows or groups.")
    for name, indices in (("training", train), ("test", test)):
        if len(indices) < 2 or data[TARGET].iloc[indices].nunique() < 2:
            raise ValueError(f"The {name} partition needs at least two rows and distinct targets.")
    empty = [name for name in FEATURES if data[name].iloc[train].isna().all()]
    if empty:
        raise ValueError(f"Training partition has all-missing features: {', '.join(empty)}")
    return train, test


def make_model(seed: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            tree_method="hist",
            device="cpu",
            n_jobs=1,
            random_state=seed,
        )),
    ])


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    if actual.shape != predicted.shape or not np.isfinite(predicted).all():
        raise ValueError("Predictions must be finite and match the target shape.")
    return {
        "n": len(actual),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(root_mean_squared_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted, force_finite=False))
        if np.unique(actual).size > 1 else None,
    }


def component_diagnostic(data: pd.DataFrame, test: np.ndarray) -> dict:
    result = {
        "definition": "Unadjusted sum of the nine sensory features plus Overall",
        "fitted": False,
        "extra_column_not_given_to_model": "Overall",
        "test_rows": len(test),
    }
    if "Overall" not in data:
        return {**result, "status": "unavailable", "reason": "Overall column is absent."}
    held_out = data.iloc[test]
    components = held_out.loc[:, [*SENSORY_FEATURES, "Overall"]]
    complete = components.notna().all(axis=1)
    result["excluded_incomplete_rows"] = int((~complete).sum())
    if not complete.any():
        return {**result, "status": "unavailable", "reason": "No complete held-out component rows."}
    actual = held_out.loc[complete, TARGET].to_numpy()
    predicted = components.loc[complete].sum(axis=1).to_numpy()
    residuals = np.abs(actual - predicted)
    return {
        **result,
        "status": "measured",
        "metrics": metrics(actual, predicted),
        "max_absolute_residual": float(residuals.max()),
        "tolerance_points": 0.05,
        "rows_within_tolerance": int((residuals <= 0.05 + 1e-12).sum()),
    }


def positions_hash(indices: np.ndarray) -> str:
    return hashlib.sha256(
        json.dumps(indices.tolist(), separators=(",", ":")).encode("ascii")
    ).hexdigest()


def evaluate(path: Path, seed: int = 42, test_size: float = 0.25) -> dict:
    started = time.perf_counter()
    data, source = load_data(path)
    groups, group_audit = build_groups(data)
    train, test = split_data(data, groups, seed, test_size)
    features, target = data.loc[:, FEATURES], data[TARGET].to_numpy()
    model = make_model(seed)
    baseline = DummyRegressor(strategy="mean")
    with threadpool_limits(limits=1):
        model.fit(features.iloc[train], target[train])
        baseline.fit(np.zeros((len(train), 1)), target[train])
        scores = {}
        for name, indices in (("train", train), ("test", test)):
            scores[name] = {
                "xgboost": metrics(target[indices], model.predict(features.iloc[indices])),
                "training_mean": metrics(
                    target[indices], baseline.predict(np.zeros((len(indices), 1)))
                ),
            }
    # sklearn represents XGBoost's missing-value default as NaN; JSON must be strict.
    parameters = model.named_steps["regressor"].get_params()
    parameters["missing"] = "NaN"
    return {
        "schema_version": 1,
        "task": "regression",
        "scope": "retrospective_cupping_score_estimation_not_pre_cupping_forecasting",
        "dataset": source,
        "target": TARGET,
        "features": list(FEATURES),
        "group_audit": group_audit,
        "split": {
            "strategy": "GroupShuffleSplit over connected country/farm and exact input-profile groups",
            "seed": seed,
            "requested_test_group_fraction": test_size,
            "actual_test_row_fraction": len(test) / len(data),
            "train_rows": len(train),
            "test_rows": len(test),
            "train_groups": len(np.unique(groups[train])),
            "test_groups": len(np.unique(groups[test])),
            "row_overlap": 0,
            "group_overlap": 0,
            "train_positions_sha256": positions_hash(train),
            "test_positions_sha256": positions_hash(test),
            "position_hash_encoding": "SHA256 of compact JSON array of zero-based CSV data-row positions",
        },
        "preprocessing": {
            "altitude": "Stateless numeric/range parsing before split; units assumed metres",
            "imputation": "Per-feature median fitted on training rows only",
            "scaling": "StandardScaler fitted on imputed training rows only",
            "feature_order": list(FEATURES),
            "train_medians": model.named_steps["imputer"].statistics_.tolist(),
            "train_scaler_mean": model.named_steps["scaler"].mean_.tolist(),
            "train_scaler_scale": model.named_steps["scaler"].scale_.tolist(),
        },
        "model": {
            "class": "xgboost.XGBRegressor",
            "fresh_fit": True,
            "parameter_selection": "Fixed legacy notebook recipe; no search or test-set tuning in this run",
            "parameters": parameters,
        },
        "baseline": {
            "class": "sklearn.dummy.DummyRegressor",
            "parameters": {"strategy": "mean"},
            "train_target_mean": float(baseline.constant_.item()),
        },
        "metrics": scores,
        "component_sum_diagnostic": component_diagnostic(data, test),
        "limitations": LIMITATIONS,
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "system": platform.system(),
            "machine": platform.machine(),
            "packages": {
                **{name: version(name) for name in (
                    "numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl",
                )},
                "xgboost": xgboost.__version__,
            },
            "thread_limit": 1,
            "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "elapsed_seconds": time.perf_counter() - started,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Existing local CQI-format CSV; never downloaded")
    parser.add_argument("--output", type=Path, required=True, help="New JSON report path; existing files are not overwritten")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25, help="Fraction of groups, not rows")
    args = parser.parse_args()
    try:
        if args.output.exists() or args.output.resolve() == args.data.resolve():
            raise ValueError("Output must be a new file, separate from the input CSV.")
        report = evaluate(args.data, args.seed, args.test_size)
        serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as destination:
            destination.write(serialized)
    except (OSError, ValueError, UnicodeError, csv.Error) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 2
    print(f"Local evaluation report written to {args.output}; provenance remains unverified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
