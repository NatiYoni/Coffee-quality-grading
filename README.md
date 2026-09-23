# Coffee Quality Intelligence
### Machine Learning System for Ethiopian Coffee Quality Prediction, Grading & Analysis

> Predicts specialty coffee quality scores from cupping data, classifies roast levels and defect types from bean imagery, and attaches SHAP feature attributions. An optional Gemini integration selects evidence-backed explanation points, with server-side validation and plain-language rendering.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Project Structure](#3-project-structure)
4. [ML Models](#4-ml-models)
5. [Quick Start](#5-quick-start)
6. [API Reference](#6-api-reference)
7. [Frontend Pages](#7-frontend-pages)
8. [Environment Variables](#8-environment-variables)
9. [Results & Performance](#9-results--performance)
10. [Limitations & Evaluation Notes](#10-limitations--evaluation-notes)
11. [Testing](#11-testing)
12. [Tech Stack](#12-tech-stack)
13. [Roadmap](#13-roadmap)
14. [Contributors](#14-contributors)
15. [References](#15-references)

---

## 1. Project Overview

Ethiopia is the birthplace of Arabica coffee and one of the world's most significant specialty coffee-producing nations. Quality assessment in the supply chain still relies heavily on manual cupping and subjective grading — a process that is slow, inconsistent, and hard to scale.

This project applies machine learning to automate and standardize three core grading tasks:

| Task | Approach | Result |
|------|----------|--------|
| Quality score prediction | Linear Regression baseline vs. XGBoost + SHAP explainability | Test R² = 0.994 (linear) / 0.960 (XGBoost) — see [§10](#10-limitations--evaluation-notes) |
| Roast level classification | MobileNetV2 CNN (transfer learning) | 99.5% on a 400-image set that was also used for validation — see [§10](#10-limitations--evaluation-notes) |
| Bean defect classification | MobileNetV2 CNN (9 defect classes) | Held-out test accuracy = 87% |
| Flavor profile segmentation | K-Means clustering (k = 4) | 4 sensory clusters |

A **Next.js + Go + FastAPI** full-stack application exposes the models through a web interface, with per-prediction SHAP feature attributions, on-demand evidence-checked explanations, side-by-side sample comparison, and an Ethiopian regional quality explorer.

---

## 2. System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Next.js       │────▶│   Go Backend    │────▶│   FastAPI       │
│   Frontend      │     │   (port 8000)   │     │   (port 8001)   │
│   (6 pages)     │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                               ┌────────┴────────┐
                                               │   ML Models     │
                                               │  · XGBoost      │
                                               │  · K-Means      │
                                               │  · CNN (roast)  │
                                               │  · CNN (defect) │
                                               │  · SHAP         │
                                               └─────────────────┘
```

**Request flow:**
1. User interacts with the **Next.js** frontend (port 3000)
2. The frontend calls the **Go backend** REST API (port 8000)
3. Go proxies ML inference requests to the **FastAPI microservice** (port 8001)
4. FastAPI loads trained models and returns predictions + SHAP values
5. Go forwards the ML service response to the frontend, preserving success and error status codes for prediction requests.
6. Only when the user clicks **Explain this prediction**, FastAPI recomputes the evidence, requests a structured explanation plan from Gemini, validates it, and renders supported statements. No Gemini call is made by `/predict`.

---

## 3. Project Structure

```
coffee-quality-grading/
├── notebooks/
│   ├── coffee_quality_model.ipynb   # EDA → Linear Regression → XGBoost → K-Means → SHAP
│   ├── roast_classifier.ipynb       # MobileNetV2 transfer learning (roast level)
│   ├── defect_detector.ipynb        # MobileNetV2 transfer learning (defect type)
│   ├── results*.txt                 # Training logs — the source of every metric in this README
│   └── outputs/models/              # Exported .pkl / .h5 / .json artifacts
│
├── ml_service/
│   ├── main.py                      # FastAPI app (port 8001)
│   ├── explanations.py              # Gemini client, evidence checks and safe text rendering
│   ├── evaluate_explanations.py      # Offline guardrail and opt-in live generation evaluation
│   ├── evals/                       # Synthetic evidence and adversarial explanation cases
│   ├── models/                      # Artifacts loaded by the inference service (Git LFS)
│   ├── tests/                       # pytest suite for the API layer (see §11)
│   ├── requirements.txt             # Runtime deps (TensorFlow, XGBoost, SHAP, …)
│   ├── requirements-test.txt        # Light test deps (no TF / XGBoost / SHAP)
│   └── Dockerfile                   # Hugging Face Spaces image (port 7860)
│
├── backend/
│   ├── main.go                      # Go HTTP server (port 8000) — Gin framework
│   ├── go.mod
│   └── .env.example
│
├── frontend/
│   ├── app/
│   │   ├── layout.js
│   │   ├── page.js                  # Landing page
│   │   ├── predict/page.js          # Quality predictor form + SHAP output
│   │   ├── predict/roast/page.js    # Roast level image classifier
│   │   ├── predict/defect/page.js   # Defect type image classifier
│   │   ├── compare/page.js          # Side-by-side two-coffee comparison
│   │   └── explore/page.js          # Ethiopian regional quality explorer
│   ├── package.json
│   ├── next.config.js
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   └── app/globals.css
│
├── coffee-bean-defect/              # Defect image dataset (Roboflow Universe, CC BY 4.0)
├── documentation/                   # Technical report (.docx) and presentation (.pptx)
├── .github/workflows/               # CI: Python evaluation, Go tests and browser checks
└── README.md
```

---

## 4. ML Models

### 4.1 Quality Scorer — XGBoost Regression

Predicts the **Total Cup Points** (SCA specialty coffee score, 0–100) from cupping sub-scores and physical attributes.

| Property | Detail |
|----------|--------|
| Target variable | `Total.Cup.Points` (range 78–89.3, mean 83.71) |
| Training samples | 207 rows (155 train / 52 test) |
| Features (13) | Altitude, Aroma, Flavor, Aftertaste, Acidity, Body, Balance, Uniformity, Clean Cup, Sweetness, Moisture Percentage, Category One Defects, Category Two Defects |
| Baseline | Linear Regression (Test R² = 0.994, RMSE = 0.128; 5-fold CV R² = 0.992) |
| Served model | XGBoost with GridSearchCV — `max_depth=3`, `n_estimators=300`, `learning_rate=0.05` (Test R² = 0.960, RMSE = 0.341; 5-fold CV R² = 0.958) |
| Explainability | SHAP `TreeExplainer` — per-prediction feature impact (top 4 by magnitude, with sign) |

**Key finding:** Linear Regression outperforms XGBoost on this dataset. That is expected rather than surprising: the SCA Total Cup Points score is, by construction, close to the *sum* of the sensory sub-scores that are fed in as features, so the target is nearly a linear function of the inputs and a linear model can reproduce it almost exactly. XGBoost shows mild overfitting at this dataset size (train R² 0.999 vs. test 0.960). For the same reason, the high R² values should **not** be read as evidence of strong predictive power — see [§10](#10-limitations--evaluation-notes). The SHAP analysis confirms Flavor (r = 0.94), Aftertaste (r = 0.93), and Balance (r = 0.93) as the dominant quality drivers.

### 4.2 Roast Level Classifier — MobileNetV2 CNN

Classifies coffee bean images into one of four roast levels for automated visual quality control.

| Property | Detail |
|----------|--------|
| Classes | `Dark`, `Green` (unroasted), `Light`, `Medium` |
| Dataset | 1,600 images (1,200 train / 400 validation), 224×224 px |
| Architecture | MobileNetV2 (ImageNet pretrained, frozen) → GlobalAvgPool2D → Dense(128, ReLU) → Dense(4, Softmax) |
| Optimizer / Loss | Adam / Categorical Cross-Entropy |
| Epochs | 10 |
| Best val accuracy | **100%** (Epoch 8) |
| Final accuracy on the 400-image set | **99.5%** — 2 misclassifications out of 400 |

> ⚠️ **Evaluation caveat:** the Kaggle dataset ships with `train/` and `test/` folders only. The `test/` folder was used as `validation_data` during training *and* for the 99.5% figure above, so that number is optimistic (it was indirectly used for model selection). A three-way split or cross-validation is on the [roadmap](#13-roadmap).

### 4.3 Defect Classifier — MobileNetV2 CNN (9 Classes)

Identifies the type of bean defect from an image — the classes map to SCA primary and secondary defect categories. This is an image *classifier* (one label + confidence per image), not an object detector: it does not localise defects or draw bounding boxes.

| Property | Detail |
|----------|--------|
| Classes (9) | `black`, `broken`, `foreign`, `fraghusk`, `green`, `husk`, `immature`, `infested`, `sour` |
| Dataset | 1,260 images (1,005 train / 123 val / 132 test), 224×224 px |
| Architecture | MobileNetV2 (frozen) → GlobalAvgPool2D → Dense(128, ReLU) → Dropout(0.3) → Dense(9, Softmax) |
| Callbacks | ModelCheckpoint (best `val_accuracy`) + EarlyStopping (patience = 5) |
| Best val accuracy | **85.37%** (Epoch 11) |
| Final test accuracy | **87%** (weighted F1 = 0.86) |

> ⚠️ **Note:** Classes with very low test support (`sour`: 3 samples, `immature`: 4, `husk`: 5) show degraded F1 scores. Additional data collection is the highest-priority improvement for this model. The dataset also contains no healthy-bean class, so the model always returns one of the nine defect types (see [§10](#10-limitations--evaluation-notes)).

### 4.4 Flavor Clustering — K-Means

Unsupervised segmentation of coffees into flavor profiles for market positioning and blending.

| Property | Detail |
|----------|--------|
| Algorithm | K-Means |
| Features | Aroma, Flavor, Acidity, Body, Balance |
| Optimal k | 4 (elbow method, final inertia = 20.88) |

| Cluster | Label | Aroma | Flavor | Acidity | Body | Balance |
|---------|-------|-------|--------|---------|------|---------|
| 0 | Bright & Floral | 8.17 | 8.19 | 8.07 | 7.98 | 8.05 |
| 1 | Earthy & Full-bodied | 7.64 | 7.67 | 7.63 | 7.60 | 7.59 |
| 2 | Balanced & Sweet | 7.89 | 7.92 | 7.85 | 7.77 | 7.79 |
| 3 | Mild & Neutral | 7.35 | 7.35 | 7.33 | 7.32 | 7.28 |

---

## 5. Quick Start

### Prerequisites

- Python 3.10+ (Anaconda recommended)
- Go 1.21+
- Node.js 18+
- Kaggle account (for dataset download)

---

### Step 1 — Set up Python environment

```bash
conda create -n coffee-ml python=3.10
conda activate coffee-ml
conda install pandas numpy matplotlib seaborn scikit-learn jupyter
pip install xgboost shap fastapi uvicorn kagglehub joblib tensorflow
```

### Step 2 — Train models (Jupyter Notebooks)

Run the three notebooks from the `notebooks/` directory:

```bash
cd notebooks
jupyter notebook
```

| Notebook | What it does | Output |
|----------|-------------|--------|
| `coffee_quality_model.ipynb` | EDA → Linear Regression → XGBoost (GridSearchCV) → K-Means → SHAP | `.pkl` / `.json` files in `notebooks/outputs/models/` |
| `roast_classifier.ipynb` | MobileNetV2 transfer learning on bean imagery (Kaggle) | `roast_classifier.h5` + `roast_classes.json` |
| `defect_detector.ipynb` | MobileNetV2 transfer learning on the Roboflow Universe defect dataset (committed under `coffee-bean-defect/`) | `defect_detector.h5` + `defect_classes.json` |

All exported artifacts land in `notebooks/outputs/models/`; the copies served by the API live in `ml_service/models/` (tracked with Git LFS — run `git lfs pull` after cloning).

### Step 3 — Start the FastAPI ML microservice

```bash
cd ml_service
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001
```

Verify it's running: `curl http://localhost:8001/health`

Gemini is optional. To enable the explanation button, follow the ML-service configuration in [§8](#8-environment-variables). Never put an API key in frontend code.

### Step 4 — Start the Go backend

```bash
cd backend
cp .env.example .env
# Adjust PYTHON_SERVICE_URL / PORT if needed
go mod tidy
go run main.go
```

Verify it's running: `curl http://localhost:8000/api/health`

### Step 5 — Start the Next.js frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

---

## 6. API Reference

### Go Backend — port 8000

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/predict` | Predict quality score from cupping features; returns score + SHAP + cluster |
| `POST` | `/api/explain` | Generate an evidence-checked explanation for a complete numeric sample |
| `POST` | `/api/compare` | Side-by-side comparison of two coffee samples |
| `POST` | `/api/predict-roast` | Classify roast level from uploaded bean image |
| `POST` | `/api/predict-defect` | Classify the defect type in an uploaded bean image |
| `GET`  | `/api/regions` | Ethiopian regional quality data and rankings |
| `GET`  | `/api/health` | Backend + ML service health check |

### FastAPI ML Service — port 8001

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict` | XGBoost inference + SHAP values + K-Means cluster assignment |
| `POST` | `/explain` | Recompute prediction evidence, validate Gemini's plan, render supported statements |
| `POST` | `/predict-roast` | CNN roast level classification |
| `POST` | `/predict-defect` | CNN defect-type classification (top class + confidence) |
| `GET`  | `/regions` | Precomputed regional quality statistics |
| `GET`  | `/health` | Model loading status for all models |

Error semantics of the ML service: `400` — image payload is not valid base64 or not a decodable image · `422` — request body is malformed or one of the required sensory features (`Aroma`, `Flavor`, `Acidity`, `Body`, `Balance`) is missing · `503` — models are not loaded yet. Any other training feature that is omitted from `/predict` is zero-filled in training column order (see [§10](#10-limitations--evaluation-notes)).

#### Example — Quality Prediction Request

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "Aroma": 8.17, "Flavor": 8.19, "Aftertaste": 8.08,
      "Acidity": 8.07, "Body": 7.98, "Balance": 8.05,
      "Uniformity": 10, "Clean Cup": 10, "Sweetness": 10,
      "Altitude": 2000, "Moisture Percentage": 11.5,
      "Category One Defects": 0, "Category Two Defects": 2
    }
  }'
```

The response contains `score`, `grade` (`Specialty` at ≥ 80 points, otherwise `Below Specialty`), `shap` (top-4 feature attributions with `direction`), `flavorCluster` (`id`, `name`, `description`) and `counterfactual` (a hint, only for below-specialty coffees).

### Evidence-checked explanations

Send the same numeric `features` object to `POST /api/explain`. Unlike `/predict`, this endpoint requires **all 13 training features**, finite JSON numbers, and no additional fields. It does not accept a client-supplied score, SHAP values, or a free-text prompt.

1. The ML service recomputes the score, grade, and top SHAP attributions from the submitted features.
2. Gemini receives only that derived evidence. It chooses the order, supported features, directions, and one of two approved wording styles. It cannot supply numbers, grades, advice, or arbitrary prose.
3. The validator rejects unknown or duplicate features, reversed contribution directions, additional fields, and omission of the strongest positive or negative driver when present.
4. The server renders sentences using the actual evidence values. The response includes `method: "validated_evidence_plan"`, the requested `model`, `sentences` with `evidence_id` citations, the source `evidence`, and explanatory `notes`.

This is **constrained, LLM-assisted explanation planning**, not free-form generation, RAG, or an agent. The restricted output makes the displayed facts checkable; it does not establish the accuracy of the underlying ML models or turn SHAP associations into causal recommendations.

The browser shows supporting evidence, caches the explanation only for the current result, and clears it when inputs change. If the recomputed evidence differs from the displayed prediction, it asks for a new prediction instead of showing an explanation for the wrong result.

| Status | Meaning |
|--------|---------|
| `422` | Incomplete, nonnumeric, nonfinite, or unexpected input |
| `429` | Provider quota exhausted; a numeric `Retry-After` is forwarded when available |
| `502` | Provider unavailable, malformed/incomplete output, or a rejected evidence plan |
| `503` | Models unavailable, no Gemini key, or rejected key/model configuration |
| `504` | Provider or ML-service timeout |

The Go proxy preserves these errors. The ML service keeps one pooled HTTPS connection to Gemini and retries a request exactly once, only when Gemini provably never processed it (connection failure or an immediate `503`); read timeouts, quota errors, and rejected plans are never retried, and there is no hardcoded "AI" fallback. Prediction results remain available if explanations fail. Logs contain the outcome code, requested model, and latency, not prompts, sample values, keys, or provider response bodies.

---

## 7. Frontend Pages

| Route | Description |
|-------|-------------|
| `/` | Landing page — hero, how-it-works flow, feature grid |
| `/predict` | Quality predictor form with SHAP attributions and optional Gemini-assisted explanations |
| `/predict/roast` | Upload a bean photo → roast level prediction with confidence |
| `/predict/defect` | Upload a bean photo → predicted defect type with confidence |
| `/compare` | Side-by-side comparison of two coffee samples across all dimensions |
| `/explore` | Ethiopian regional quality rankings, maps, and score distributions |

---

## 8. Environment Variables

### Backend — `backend/.env`

```env
PYTHON_SERVICE_URL=http://localhost:8001
PORT=8000
```

### ML service — `ml_service/.env` (optional)

Create a key in [Google AI Studio](https://aistudio.google.com/apikey). To stay within free access, use a project **without billing enabled** and a model currently eligible for a free tier; check [pricing](https://ai.google.dev/gemini-api/docs/pricing) and your account's quotas. The model is configurable because availability and limits change.

```bash
cd ml_service
cp .env.example .env
# Edit .env locally; never paste or commit the key.
uvicorn main:app --host 127.0.0.1 --port 8001 --env-file .env
```

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.6-flash
```

The integration calls [Gemini's OpenAI-compatible endpoint](https://ai.google.dev/gemini-api/docs/openai) directly through `httpx`; an OpenAI account or subscription is not needed. `gemini-3.6-flash` is the default because it is the model Google's API currently points new users to and the one the live evaluation below was recorded with; if Gemini answers `503` under load, another `*-flash` model listed for your key can be set without code changes. Without a key, ordinary ML predictions still work and `/explain` reports `503`.

**Privacy and deployment:** Google may use free-tier inputs and outputs to improve its products. Use public or synthetic coffee samples only. `.env` files are excluded from Git and the ML Docker build context; configure deployment secrets at runtime instead. Keep a public demo behind suitable access and rate controls: this project does not provide authentication or distributed abuse prevention, and provider quota errors are not a substitute for those controls.

### Frontend — `frontend/.env.local`

```env
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

---

## 9. Results & Performance

### Model Summary

| Model | Type | Key Metric | Score | Notes |
|-------|------|-----------|-------|-------|
| Quality Scorer (Linear Regression) | Regression | Test R² | 0.994 | Best model on this data; not the one currently served |
| Quality Scorer (XGBoost) | Regression | Test R² | 0.960 | Served model; mild overfitting (train R² 0.999) |
| Roast Classifier | CNN — 4 class | Accuracy (validation set) | 99.5% | Optimistic — no separate held-out test set |
| Defect Classifier | CNN — 9 class | Held-out test accuracy | 87% | Weak on `sour` / `immature` / `husk` (3–5 test images each) |
| Flavor Clustering | Unsupervised | Clusters (k) | 4 | k chosen by elbow method; cluster names assigned by hand |

### Ethiopian Regional Quality

All tested Ethiopian regions exceed the SCA Specialty Grade threshold of 80 points:

| Region | Avg Predicted Score | Notes |
|--------|-------------------|-------|
| Guji | ~86.0 | Highest — complex fruit-forward profiles |
| OROMIA | ~84.5 | Consistently above specialty grade |
| Sidama | ~84.0 | Reliable specialty grade |

---

## 10. Limitations & Evaluation Notes

Stated plainly, so that nobody has to discover them in the notebooks:

1. **The quality-score target is almost additive in its inputs.** SCA Total Cup Points is essentially the sum of ten sensory sub-scores minus defect penalties, and nine of those sub-scores are model inputs. A linear model therefore reaches R² ≈ 0.99 largely by re-adding its inputs. The regression results demonstrate a correct pipeline (split, cross-validation, tuning, SHAP) much more than they demonstrate predictive power. Predicting the score from *non-sensory* inputs only (altitude, region, processing, moisture, defects) is the genuinely hard problem and is on the roadmap.
2. **Small tabular dataset.** 207 usable rows (155 train / 52 test); confidence intervals on the test metrics are wide.
3. **Roast classifier: the validation set is the reported test set.** See §4.2. Treat 99.5% as an upper bound.
4. **The defect classifier has no "no defect" class.** The dataset contains only defective beans, so the model always returns one of the nine defect types and the API's `isDefective` flag is always `true` — the frontend's "No Defects Found" state is currently unreachable. Minority classes (`sour`, `immature`, `husk`) have 3–5 test images each, so their per-class F1 is unreliable.
5. **Image models were trained on curated datasets.** Phone photos under uneven lighting are out-of-distribution; expect lower accuracy than reported.
6. **Missing features are zero-filled.** `/predict` requires `Aroma`, `Flavor`, `Acidity`, `Body`, `Balance` and fills any other omitted training column with `0` — a value the model never saw for fields such as `Altitude` or `Uniformity`. Supply all 13 features for meaningful predictions.
7. **Explanations use a constrained evidence plan.** Gemini selects points and approved wording styles; the server renders the prose from checked evidence. This limits unsupported additions, but does not fix model accuracy, validate farming interventions, or provide a conversational assistant.

---

## 11. Testing

The FastAPI suite in [`ml_service/tests/`](ml_service/tests/) uses model fakes and a mocked Gemini HTTP transport. It covers the original prediction and image contracts, explanation input validation, plan rejection, numeric rendering, refusal/truncation handling, timeouts, missing credentials, and quota propagation. No API key, TensorFlow runtime, or Git-LFS artifact is required for these tests.

```bash
cd ml_service
pip install -r requirements-test.txt
pytest -q
python evaluate_explanations.py
```

The offline evaluator uses [`evals/explanation_cases.json`](ml_service/evals/explanation_cases.json): six synthetic evidence scenarios and 21 labeled acceptable/adversarial plans. It reports verdict accuracy, false acceptances, and false rejections, and exits nonzero on **any** mismatch. This measures the validator, not Gemini's generation quality.

To evaluate the real provider separately, configure an ignored `.env` and explicitly opt in:

```bash
# From ml_service; one synthetic case for a small live smoke test:
python evaluate_explanations.py --live --env-file .env --case mixed_signs

# Omit --case to evaluate all six scenarios; this consumes free-tier requests.
python evaluate_explanations.py --live --env-file .env
```

Live mode reports the valid-plan rate and median latency, includes only accepted rendered explanations, and exits nonzero if any case fails or is unattempted. It stops after a quota error rather than retrying automatically. Neither live success nor offline acceptance is a claim that the base ML predictions are correct.

Latest recorded live run (`gemini-3.6-flash`, free tier, from a high-latency network): 6/6 valid plans, median latency 3.0 s. An earlier run of the same day scored 5/6 because one connection to Google exceeded the then 5 s connect timeout; that failure motivated the pooled connection, the 10 s connect budget, and the single connect/503 retry described above. Live results vary with provider load and are not part of CI.

Go proxy tests and browser tests:

```bash
cd backend
go test ./...

# From the repository root, in a separate shell:
cd frontend
npm ci
npx playwright install chromium
npm run build
npm run test:e2e
```

Browser tests mock the prediction/explanation HTTP endpoints and exercise opt-in generation, evidence links, error/retry states, stale results, and mobile layout. They do not make Gemini calls.

[CI](.github/workflows/ml-service-tests.yml) runs Python tests plus offline evaluation on Python 3.11, Go tests, and frontend build/browser checks for relevant pushes and pull requests. The offline JSON report is uploaded as a workflow artifact; CI never needs a Gemini key.

Not covered by these checks: notebook retraining, real model accuracy, or an end-to-end run combining the real trained artifacts with a live Gemini call.

---

## 12. Tech Stack

| Layer | Technologies |
|-------|-------------|
| **ML** | scikit-learn, XGBoost, SHAP, K-Means, TensorFlow / Keras, MobileNetV2 |
| **ML Service** | FastAPI, Uvicorn, joblib, httpx, Pydantic, pytest |
| **Backend** | Go, Gin framework |
| **Frontend** | Next.js 14, React 18, Tailwind CSS, Lucide icons, Playwright |
| **Data Sources** | Kaggle (CQI cupping data, coffee bean images), Roboflow Universe (defect images) |
| **Explainability** | SHAP TreeExplainer, Gemini structured evidence plans, deterministic validation/rendering |

---

## 13. Roadmap

- [ ] Serve the linear baseline (or both models) — it is currently the better quality-score model
- [ ] Re-evaluate the roast classifier on data that was not used for model selection
- [ ] Add a healthy-bean class so `isDefective` can actually be `false`; collect more `sour`, `immature`, and `husk` images
- [ ] Predict Total Cup Points from non-sensory features only (the useful, hard version of the problem)
- [ ] Expand live explanation evaluation across model versions and repeated runs
- [ ] An end-to-end smoke test combining the real trained artifacts and Gemini

---

## 14. Contributors

Team project. Commit history: [@NatiYoni](https://github.com/NatiYoni) (ML pipeline and model export, FastAPI inference service, tests), [@ellay21](https://github.com/ellay21), [@Natthy2023](https://github.com/Natthy2023).

---

## 15. References

1. Mitchell, M., Wu, S., Zaldivar, A., et al. (2019). *Model Cards for Model Reporting.* Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT\*'19).

2. Gebru, T., Morgenstern, J., Vecchione, B., et al. (2018). *Datasheets for Datasets.* arXiv:1803.09010.

3. fatihb. *Coffee Quality Data (CQI).* Kaggle. https://www.kaggle.com/datasets/fatihb/coffee-quality-data-cqi

4. gpiosenka. *Coffee Bean Dataset (Resized 224 x 224).* Kaggle. https://www.kaggle.com/datasets/gpiosenka/coffee-bean-dataset-resized-224-x-224

5. vishwesh-pbvmh. *coffee-bean-defect Dataset, v1 (2024-03-10).* Roboflow Universe. https://universe.roboflow.com/vishwesh-pbvmh/coffee-bean-defect-smvw1. License: CC BY 4.0.

6. Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions.* Advances in Neural Information Processing Systems (NeurIPS).

7. Sandler, M., Howard, A., Zhu, M., et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

8. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.

---

*Coffee Quality Intelligence — ML Technical Documentation v1.0*