# Local coffee-quality evaluation

**Current evidence: an evaluator and synthetic-fixture checks, not a new coffee
benchmark.** No raw quality CSV is tracked or available in this checkout. No real
data report, new performance number, trained weight, or source row is committed.
The notebook's 207 rows and 155/52 split are recorded historical output, not counts
from a new run. Tests deliberately use generated fixtures and cannot establish
model quality.

## Source and target boundary

`notebooks/coffee_quality_model.ipynb` calls
`kagglehub.dataset_download("fatihb/coffee-quality-data-cqi")`, selects the first
CSV, and does not pin a dataset version or record its hash. Its recorded schema has
41 columns and a continuous **`Total Cup Points`** target: this is regression,
not classification. Do not reuse the saved `.pkl` files as held-out evidence.

The [upstream metadata API](https://www.kaggle.com/api/v1/datasets/view/fatihb/coffee-quality-data-cqi),
inspected during this work on 2026-10-03, reports version 1, last updated
2023-05-12, total size 112,964 bytes, and license name "Other (specified in
description)". Its description does not provide a clear reuse grant. No dataset
was downloaded or redistributed for this change; supply only an existing CSV you
are authorized to use. This metadata does not authenticate a later local file.

The upstream description explicitly warns that Total Cup Points sums ten
components, although its prose enumerates nine. The served manifest
`ml_service/models/feature_columns.json` supplies these thirteen numeric inputs:

```text
Altitude, Aroma, Flavor, Aftertaste, Acidity, Body, Balance, Uniformity,
Clean Cup, Sweetness, Moisture Percentage, Category One Defects,
Category Two Defects
```

Nine are sensory score components. Even a disjoint split measures **retrospective
cupping-score estimation**, not forecasting an unknown score before cupping.
`Overall` is excluded from the model. If present and complete, it is used only in
a separately labeled, unfitted **unadjusted component-sum diagnostic** alongside
the nine sensory components. The report measures residuals; it does not assume an
exact formula, infer defect penalties, calibrate a reconstruction on held-out
labels, or call this extra-information diagnostic a fair model baseline.

## Clean-checkout setup and commands

Run from the repository root. Python 3.13 is the reference environment. The
isolated requirements use the same scikit-learn/XGBoost versions as the service,
but Python-3.13-compatible NumPy/pandas and a CPU-only XGBoost wheel. They do not
install TensorFlow, SHAP, Kaggle clients, FastAPI, or LLM packages. Package
installation requires registry access; evaluation itself is entirely local.

```bash
python3 -m venv .venv-evaluation
.venv-evaluation/bin/python -m pip install -r evaluation/requirements.txt
.venv-evaluation/bin/python -m unittest discover -s evaluation -p 'test_*.py' -v
```

To evaluate an **authorized, existing** source CSV, place an unchanged copy at
`data/coffee-quality/source.csv`. That directory is ignored to avoid accidentally
committing producer/contact information. There is intentionally no downloader or
synthetic-data fallback.

```bash
.venv-evaluation/bin/python -m evaluation.evaluate \
  --data data/coffee-quality/source.csv \
  --output .evaluation-output/report.json
```

Without that CSV, the command exits nonzero with `Evaluation failed` and writes
no report. It is not currently possible to reproduce a real-data result from this
checkout alone. Missing dependencies also fail rather than substitute a model.
Report paths must be new; existing outputs and input files are never overwritten.

Repeat with the same CSV and environment, then compare all fields except elapsed
time. Use unused output paths:

```bash
.venv-evaluation/bin/python -m evaluation.evaluate \
  --data data/coffee-quality/source.csv \
  --output .evaluation-output/repeat.json
.venv-evaluation/bin/python - <<'PY'
import json
from pathlib import Path
reports = [
    json.loads(Path(path).read_text())
    for path in (".evaluation-output/report.json", ".evaluation-output/repeat.json")
]
for report in reports:
    report["runtime"].pop("elapsed_seconds")
assert reports[0] == reports[1], "Seeded evaluation changed"
print("Reports match except elapsed time")
PY
```

Bitwise reproducibility is checked within the same runtime, dependency versions,
file bytes/order, and evaluator source, not promised across platforms or versions.
The report records resolved numerical dependency versions; the manifest pins
direct evaluation dependencies plus the resolved SciPy/joblib versions.

## What the command does

1. Reads one local UTF-8 CSV and hashes its exact bytes before parsing. Required
   headers are the thirteen inputs, target, `Country of Origin`, and `Farm Name`.
   No rows are dropped. Invalid/missing targets, malformed numeric data, duplicate
   headers, malformed records, impossible score/count ranges, and unusable splits
   fail explicitly. Missing feature values are allowed only when the training
   partition contains an observed value for that feature. Altitude range parsing
   is stateless and assumes metres; unsupported text is an error, not a default.
2. Links rows sharing normalized country/farm names **or identical parsed
   thirteen-feature profiles**, including transitive links. Targets do not choose
   the groups. Duplicate full pandas-parsed rows (before feature normalization),
   repeated input profiles, conflicting targets within those profiles, unknown
   farm names, and group sizes are counted without emitting identifiers. There are
   no actual duplicate/group counts yet because the source CSV is unavailable.
3. Applies `GroupShuffleSplit` with seed **42** and **25% of groups** held out.
   Actual row counts/fraction may differ. Both partitions must have at least two
   rows and distinct targets. Fits median imputation, then `StandardScaler`, on
   training rows only; the test partition never fits preprocessing or estimators.
   Unlike the notebook, it does not impute altitude using all rows before splitting.
4. Fits a new CPU `XGBRegressor` with 300 trees, depth 3, learning rate 0.05,
   squared-error objective, histogram tree method, and one thread. These are fixed
   legacy notebook settings, not a grid search. Compares against a training-mean
   `DummyRegressor` on the identical split. Reports train/test **MAE, RMSE (cup
   points), and R2**, never regression percentage "accuracy".
5. Writes schema-version-1 JSON with source path/hash, provenance uncertainty,
   feature/exclusion lists, missingness/group audits, seed/split counts and
   row-position hashes, fitted preprocessing statistics, estimator parameters,
   metrics, diagnostic coverage/residuals, runtime versions, evaluator hash, and
   limitations. It emits no raw rows, per-row predictions, or producer/farm/contact
   values. Diagnostic R2 is `null` when its complete-case subset cannot define it.

## Limits and interpretation

Country/farm matching is a conservative name heuristic, not verified farm or lot
identity. Case/whitespace variants are linked; aliases, near-duplicates, unknown
farms, and relationships through producers/other identifiers can still cross the
split. Identical names or profiles can also over-group genuinely different lots.
This is neither a chronological nor an unseen-country split. All countries in the
input are included; it is not an Ethiopian-only benchmark. Inspect the reported
group sizes and counts before interpreting any result.

A fresh fit prevents scoring an unknown serialized model's training rows as a new
test set. It does **not** undo earlier model-family/hyperparameter choices made
using this dataset, remove the sensory/target relationship, create an independent
external benchmark, or validate service uptime. One small split also provides no
uncertainty estimate. Poor/negative R2 must be reported rather than hidden.

`notebooks/results.txt` and the root README retain the published Linear Regression
R2 0.994 / RMSE 0.128 and XGBoost R2 0.960 / RMSE 0.341 as **historical claims**.
This evaluator does not reproduce or independently validate them. It neither
changes serving behavior nor depends on the separate, unmerged explanation work.
