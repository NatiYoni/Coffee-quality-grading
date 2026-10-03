"""Synthetic fixtures test evaluator behavior, never real coffee performance."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.evaluate import (
    FEATURES,
    SENSORY_FEATURES,
    TARGET,
    build_groups,
    component_diagnostic,
    evaluate,
    load_data,
    metrics,
    parse_altitude,
    split_data,
)

ROOT = Path(__file__).resolve().parents[1]


def fixture_data(rows: int = 32) -> pd.DataFrame:
    records = []
    for index in range(rows):
        sensory = {
            name: 6.5 + index * 0.04 + offset * 0.02
            for offset, name in enumerate(SENSORY_FEATURES[:6])
        }
        sensory.update({"Uniformity": 10.0, "Clean Cup": 10.0, "Sweetness": 10.0})
        overall = 6.8 + index * 0.035
        records.append({
            "ID": index,
            "Country of Origin": f"fixture-country-{(index // 2) % 2}",
            "Farm Name": f"fixture-private-farm-{index // 2}",
            "Producer": f"fixture-private-producer-{index}",
            "Certification Contact": f"fixture-private-contact-{index}@example.invalid",
            "Altitude": 900.0 + index * 30,
            **sensory,
            "Overall": overall,
            "Defects": 0,
            TARGET: sum(sensory.values()) + overall,
            "Moisture Percentage": 10.0 + index * 0.025,
            "Category One Defects": index % 2,
            "Category Two Defects": index % 3,
        })
    return pd.DataFrame(records)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="coffee-evaluation-fixture-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.path = self.directory / "synthetic-fixture.csv"
        self.data = fixture_data()

    def write_fixture(self, data=None):
        (self.data if data is None else data).to_csv(self.path, index=False)
        return self.path

    def cli(self, *arguments, hash_seed="0"):
        return subprocess.run(
            [sys.executable, "-m", "evaluation.evaluate", *map(str, arguments)],
            cwd=ROOT, text=True, capture_output=True, check=False,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
        )

    def test_feature_allowlist_matches_both_existing_manifests(self):
        for relative in (
            "ml_service/models/feature_columns.json",
            "notebooks/outputs/models/feature_columns.json",
        ):
            with self.subTest(manifest=relative):
                self.assertEqual(list(FEATURES), json.loads((ROOT / relative).read_text()))

    def test_stateless_altitude_parsing_and_explicit_failures(self):
        self.assertEqual(parse_altitude(" 900 - 1100 "), 1000)
        self.assertEqual(parse_altitude("1200.5"), 1200.5)
        self.assertTrue(np.isnan(parse_altitude(np.nan)))
        for invalid in ("mountain", "1-2-3", "1100-900", "-2", "inf"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_altitude(invalid)

    def test_groups_link_farms_and_duplicate_features_transitively(self):
        self.data.loc[2, list(FEATURES)] = self.data.loc[0, list(FEATURES)].to_numpy()
        self.data.loc[2, TARGET] = self.data.loc[0, TARGET] + 0.25
        self.data.loc[1, "Farm Name"] = "  FIXTURE-PRIVATE-FARM-0  "
        self.data.loc[1, "Country of Origin"] = "  FIXTURE-COUNTRY-0  "
        groups, audit = build_groups(self.data)
        self.assertEqual(len(set(groups[:4])), 1)
        self.assertNotEqual(groups[0], groups[4])
        self.assertEqual(audit["duplicate_feature_rows"], 1)
        self.assertEqual(audit["feature_profiles_with_conflicting_targets"], 1)
        self.assertEqual(audit["connected_groups"], 15)

    def test_unknown_farms_are_not_assumed_independent_of_duplicate_profiles(self):
        self.data.loc[0, "Farm Name"] = "unknown"
        self.data.loc[2, "Farm Name"] = np.nan
        self.data.loc[2, list(FEATURES)] = self.data.loc[0, list(FEATURES)].to_numpy()
        groups, audit = build_groups(self.data)
        self.assertEqual(groups[0], groups[2])
        self.assertNotEqual(groups[0], groups[1])
        self.assertEqual(audit["unknown_farm_rows"], 2)

    def test_duplicate_counts_include_exact_raw_rows(self):
        duplicated = pd.concat([self.data, self.data.iloc[[0]]], ignore_index=True)
        data, source = load_data(self.write_fixture(duplicated))
        groups, audit = build_groups(data)
        self.assertEqual(source["raw_duplicate_rows"], 1)
        self.assertEqual(audit["duplicate_feature_rows"], 1)
        self.assertEqual(groups[0], groups[-1])

    def test_split_is_reproducible_and_keeps_rows_and_groups_disjoint(self):
        data, _ = load_data(self.write_fixture())
        groups, _ = build_groups(data)
        train, test = split_data(data, groups, seed=42, test_size=0.25)
        repeated = split_data(data, groups, seed=42, test_size=0.25)
        np.testing.assert_array_equal(train, repeated[0])
        np.testing.assert_array_equal(test, repeated[1])
        self.assertFalse(set(train) & set(test))
        self.assertFalse(set(groups[train]) & set(groups[test]))
        self.assertEqual(set(train) | set(test), set(range(len(data))))
        changed = data.copy()
        changed[TARGET] += np.arange(len(data)) * 0.01
        np.testing.assert_array_equal(groups, build_groups(changed)[0])

    def test_imputation_scaling_and_estimators_never_fit_held_out_rows(self):
        self.data.loc[[3, 4], "Altitude"] = np.nan
        data, _ = load_data(self.write_fixture())
        groups, _ = build_groups(data)
        train, test = split_data(data, groups, seed=42, test_size=0.25)
        first = evaluate(self.path)
        train_features = data.loc[:, FEATURES].iloc[train]
        self.assertTrue(train_features["Altitude"].isna().any())
        self.assertTrue(data["Altitude"].iloc[test].isna().any())
        medians = train_features.median()
        np.testing.assert_allclose(first["preprocessing"]["train_medians"], medians)
        np.testing.assert_allclose(
            first["preprocessing"]["train_scaler_mean"],
            train_features.fillna(medians).mean(),
        )
        self.assertAlmostEqual(
            first["baseline"]["train_target_mean"], data[TARGET].iloc[train].mean()
        )
        self.data.loc[test, "Altitude"] = 1_000_000
        self.data.loc[test, "Aroma"] += 0.1
        self.data.loc[test, TARGET] += 0.5
        self.write_fixture()
        second = evaluate(self.path)
        self.assertEqual(first["split"], second["split"])
        self.assertEqual(first["preprocessing"], second["preprocessing"])
        self.assertEqual(first["baseline"], second["baseline"])
        self.assertEqual(first["metrics"]["train"], second["metrics"]["train"])
        self.assertNotEqual(first["metrics"]["test"], second["metrics"]["test"])

    def test_report_contract_and_no_source_identifiers(self):
        report = evaluate(self.write_fixture())
        encoded = json.dumps(report, allow_nan=False)
        self.assertEqual(json.loads(encoded), report)
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["task"], "regression")
        self.assertEqual(report["target"], TARGET)
        self.assertIn("not_pre_cupping_forecasting", report["scope"])
        self.assertEqual(report["features"], list(FEATURES))
        self.assertEqual(
            report["dataset"]["sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest()
        )
        self.assertEqual(report["dataset"]["rows"], 32)
        self.assertEqual(report["dataset"]["provenance_status"], "unverified_user_supplied_local_csv")
        self.assertIsNone(report["dataset"]["source_version"])
        self.assertIn("Producer", report["dataset"]["excluded_from_model"])
        self.assertNotIn("fixture-private", encoded)
        self.assertNotIn("@example.invalid", encoded)
        self.assertNotIn("fixture-country", encoded)
        self.assertEqual(report["model"]["parameters"]["device"], "cpu")
        self.assertEqual(report["model"]["parameters"]["n_jobs"], 1)
        self.assertEqual(report["model"]["parameters"]["n_estimators"], 300)
        self.assertEqual(report["baseline"]["parameters"], {"strategy": "mean"})
        self.assertEqual(report["split"]["train_rows"] + report["split"]["test_rows"], 32)
        self.assertEqual(report["split"]["row_overlap"], 0)
        self.assertEqual(report["split"]["group_overlap"], 0)
        for partition in ("train", "test"):
            for estimator in ("xgboost", "training_mean"):
                scores = report["metrics"][partition][estimator]
                self.assertEqual(set(scores), {"n", "mae", "rmse", "r2"})
                self.assertEqual(scores["n"], report["split"][f"{partition}_rows"])
                self.assertTrue(np.isfinite(list(scores.values())).all())
        self.assertEqual(len(report["runtime"]["evaluator_sha256"]), 64)
        self.assertIn("scikit-learn", report["runtime"]["packages"])

    def test_repeated_seeded_reports_differ_only_in_elapsed_time(self):
        self.write_fixture()
        first, second = evaluate(self.path), evaluate(self.path)
        first["runtime"].pop("elapsed_seconds")
        second["runtime"].pop("elapsed_seconds")
        self.assertEqual(first, second)

    def test_component_sum_is_a_diagnostic_not_a_predictor(self):
        test = np.array([0, 1, 2])
        diagnostic = component_diagnostic(self.data, test)
        self.assertFalse(diagnostic["fitted"])
        self.assertEqual(diagnostic["status"], "measured")
        self.assertEqual(diagnostic["rows_within_tolerance"], 3)
        self.assertAlmostEqual(diagnostic["metrics"]["rmse"], 0)
        self.data.loc[0, TARGET] += 1
        diagnostic = component_diagnostic(self.data, test)
        self.assertEqual(diagnostic["rows_within_tolerance"], 2)
        self.assertAlmostEqual(diagnostic["max_absolute_residual"], 1)

    def test_missing_components_are_explicit_and_not_imputed_for_diagnostic(self):
        test = np.array([0, 1])
        absent = component_diagnostic(self.data.drop(columns="Overall"), test)
        self.assertEqual(absent["status"], "unavailable")
        self.assertNotIn("metrics", absent)
        self.data.loc[0, "Overall"] = np.nan
        partial = component_diagnostic(self.data, test)
        self.assertEqual(partial["metrics"]["n"], 1)
        self.assertEqual(partial["excluded_incomplete_rows"], 1)
        self.assertIsNone(partial["metrics"]["r2"])
        self.data.loc[1, "Overall"] = np.nan
        empty = component_diagnostic(self.data, test)
        self.assertEqual(empty["status"], "unavailable")
        self.assertNotIn("metrics", empty)

    def test_metric_values_and_undefined_r2_are_honest(self):
        scores = metrics(np.array([1.0, 3.0]), np.array([2.0, 2.0]))
        self.assertEqual(scores, {"n": 2, "mae": 1.0, "rmse": 1.0, "r2": 0.0})
        self.assertIsNone(metrics(np.array([1.0, 1.0]), np.array([1.0, 1.0]))["r2"])
        with self.assertRaisesRegex(ValueError, "finite"):
            metrics(np.array([1.0]), np.array([np.inf]))
        with self.assertRaisesRegex(ValueError, "shape"):
            metrics(np.array([1.0]), np.array([1.0, 2.0]))

    def test_invalid_columns_and_values_fail_without_dropping_rows(self):
        mutations = (
            (TARGET, np.nan, "missing targets"),
            ("Aroma", "not-a-number", "non-numeric"),
            ("Aroma", 11, "between 0 and 10"),
            ("Body", np.inf, "infinite"),
            ("Moisture Percentage", -1, "negative"),
            ("Category One Defects", 0.5, "whole counts"),
            ("Altitude", "2000-1000", "low-high"),
            (TARGET, 101, "between 0 and 100"),
        )
        for column, value, message in mutations:
            with self.subTest(column=column, value=value):
                changed = self.data.copy()
                changed[column] = changed[column].astype(object)
                changed.loc[0, column] = value
                with self.assertRaisesRegex(ValueError, message):
                    load_data(self.write_fixture(changed))
        for column in (TARGET, "Farm Name", "Country of Origin", "Flavor"):
            with self.subTest(missing=column), self.assertRaisesRegex(ValueError, "required columns"):
                load_data(self.write_fixture(self.data.drop(columns=column)))

    def test_malformed_empty_and_missing_csv_fail(self):
        for content, message in (
            ("", "empty"),
            ("Aroma,Aroma\n1,2\n", "duplicate"),
            ("Aroma,Flavor\n1,2,3\n", "number of fields"),
        ):
            with self.subTest(content=content):
                self.path.write_text(content)
                with self.assertRaisesRegex(ValueError, message):
                    load_data(self.path)
        with self.assertRaisesRegex(ValueError, "no data rows"):
            load_data(self.write_fixture(self.data.iloc[:0]))
        with self.assertRaises(FileNotFoundError):
            load_data(self.directory / "missing.csv")

    def test_unusable_split_and_training_columns_fail_explicitly(self):
        data, _ = load_data(self.write_fixture())
        groups, _ = build_groups(data)
        for seed, size in ((-1, 0.25), (2**32, 0.25), (42, 0), (42, 1), (42, float("nan"))):
            with self.subTest(seed=seed, size=size), self.assertRaises(ValueError):
                split_data(data, groups, seed, size)
        with self.assertRaisesRegex(ValueError, "independent groups"):
            split_data(data, np.zeros(len(data)), 42, 0.25)
        constant = data.copy()
        constant[TARGET] = 80
        with self.assertRaisesRegex(ValueError, "distinct targets"):
            split_data(constant, groups, 42, 0.25)
        with self.assertRaisesRegex(ValueError, "at least two rows"):
            split_data(data.iloc[:4], np.arange(4), 42, 0.25)
        train, _ = split_data(data, groups, 42, 0.25)
        data.loc[train, "Altitude"] = np.nan
        with self.assertRaisesRegex(ValueError, "all-missing features: Altitude"):
            split_data(data, groups, 42, 0.25)

    def test_cli_writes_strict_report_and_refuses_existing_output(self):
        output = self.directory / "reports" / "fixture-only.json"
        result = self.cli("--data", self.write_fixture(), "--output", output)
        self.assertEqual(result.returncode, 0, result.stderr)
        original = output.read_bytes()
        self.assertEqual(json.loads(original)["dataset"]["rows"], 32)
        result = self.cli("--data", self.path, "--output", output)
        self.assertEqual(result.returncode, 2)
        self.assertIn("new file", result.stderr)
        self.assertEqual(output.read_bytes(), original)
        source = self.path.read_bytes()
        result = self.cli("--data", self.path, "--output", self.path)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.path.read_bytes(), source)

    def test_cli_missing_data_fails_without_report_or_synthetic_fallback(self):
        output = self.directory / "must-not-exist.json"
        result = self.cli("--data", self.directory / "missing.csv", "--output", output)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Evaluation failed", result.stderr)
        self.assertFalse(output.exists())
        self.assertEqual(result.stdout, "")

    def test_cli_reports_repeat_across_processes_and_python_hash_seeds(self):
        self.write_fixture()
        reports = []
        for hash_seed in ("1", "2"):
            output = self.directory / f"fixture-only-{hash_seed}.json"
            result = self.cli(
                "--data", self.path, "--output", output, hash_seed=hash_seed,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text())
            report["runtime"].pop("elapsed_seconds")
            reports.append(report)
        self.assertEqual(reports[0], reports[1])

    def test_missing_dependencies_fail_instead_of_publishing_results(self):
        output = self.directory / "must-not-exist.json"
        result = subprocess.run(
            [sys.executable, "-S", "-m", "evaluation.evaluate",
             "--data", str(self.write_fixture()), "--output", str(output)],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ModuleNotFoundError", result.stderr)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
