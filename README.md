# Coffee Quality Intelligence
### Machine Learning System for Ethiopian Coffee Quality Prediction, Grading & Analysis

> Predicts specialty coffee quality scores from cupping data, classifies roast levels and defects from bean imagery, segments flavor profiles via clustering, and explains every prediction in plain language — with a full-stack web application for producers, buyers, and researchers.

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
10. [Tech Stack](#10-tech-stack)
11. [Build Order](#11-build-order)
12. [Deliverables](#12-deliverables)
13. [References](#13-references)

---

## 1. Project Overview

Ethiopia is the birthplace of Arabica coffee and one of the world's most significant specialty coffee-producing nations. Quality assessment in the supply chain still relies heavily on manual cupping and subjective grading — a process that is slow, inconsistent, and hard to scale.

This project applies machine learning to automate and standardize three core grading tasks:

| Task | Approach | Result |
|------|----------|--------|
| Quality score prediction | XGBoost regression + SHAP explainability | Test R² = 0.960, RMSE = 0.341 |
| Roast level classification | MobileNetV2 CNN (transfer learning) | Test accuracy = 99.5% |
| Bean defect detection | MobileNetV2 CNN (9 defect classes) | Test accuracy = 87% |
| Flavor profile segmentation | K-Means clustering (k = 4) | 4 clean sensory clusters |

A **Next.js + Go + FastAPI** full-stack application exposes all models through a clean web interface, with per-prediction SHAP explanations in plain language, side-by-side sample comparison, and an Ethiopian regional quality explorer.

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
                                               │  · SHAP         │
                                               └─────────────────┘
```

**Request flow:**
1. User interacts with the **Next.js** frontend (port 3000)
2. The frontend calls the **Go backend** REST API (port 8000)
3. Go proxies ML inference requests to the **FastAPI microservice** (port 8001)
4. FastAPI loads trained models and returns predictions + SHAP values
5. Go formats the response and optionally enriches it via the Anthropic API (plain-language explanation)

---

## 3. Project Structure

```
coffee-quality-predictor/
├── notebooks/
│   ├── coffee_quality_model.ipynb   # XGBoost + K-Means + SHAP training
│   ├── roast_classifier.ipynb       # MobileNetV2 transfer learning
│   └── defect_detector.ipynb        # Roboflow API defect detection
│
├── outputs/
│   ├── charts/                      # EDA + training PNG exports
│   └── models/                      # Exported .pkl / .h5 / .json files
│
├── ml_service/
│   ├── main.py                      # FastAPI app (port 8001)
│   └── requirements.txt
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
│   │   ├── predict/defect/page.js   # Defect bounding-box detector
│   │   ├── compare/page.js          # Side-by-side two-coffee comparison
│   │   └── explore/page.js          # Ethiopian regional quality explorer
│   ├── package.json
│   ├── next.config.js
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   └── app/globals.css
│
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
| Features (13) | Altitude, Aroma, Flavor, Aftertaste, Acidity, Body, Balance, Uniformity, Clean Cup, Sweetness, Moisture, Defects, Processing Method |
| Baseline | Linear Regression (Test R² = 0.994, RMSE = 0.128) |
| Best model | XGBoost with GridSearchCV — `max_depth=3`, `n_estimators=300`, `learning_rate=0.05` |
| Explainability | SHAP `TreeExplainer` — per-prediction feature impact |

**Key finding:** Linear Regression outperforms XGBoost on this dataset due to the near-linear additive structure of SCA cupping sub-scores. XGBoost shows mild overfitting at this dataset size. The SHAP analysis confirms Flavor (r = 0.94), Aftertaste (r = 0.93), and Balance (r = 0.93) as the dominant quality drivers.

### 4.2 Roast Level Classifier — MobileNetV2 CNN

Classifies coffee bean images into one of four roast levels for automated visual quality control.

| Property | Detail |
|----------|--------|
| Classes | `Dark`, `Green` (unroasted), `Light`, `Medium` |
| Dataset | 1,600 images (1,200 train / 400 val-test), 224×224 px |
| Architecture | MobileNetV2 (ImageNet pretrained, frozen) → GlobalAvgPool2D → Dense(128, ReLU) → Dense(4, Softmax) |
| Optimizer / Loss | Adam / Categorical Cross-Entropy |
| Epochs | 10 |
| Best val accuracy | **100%** (Epoch 8) |
| Final test accuracy | **99.5%** — only 2 misclassifications out of 400 |

### 4.3 Defect Detector — MobileNetV2 CNN (9 Classes)

Identifies the type of bean defect from an image — maps directly to SCA primary and secondary defect grading.

| Property | Detail |
|----------|--------|
| Classes (9) | `black`, `broken`, `foreign`, `fraghusk`, `green`, `husk`, `immature`, `infested`, `sour` |
| Dataset | 1,260 images (1,005 train / 123 val / 132 test), 224×224 px |
| Architecture | MobileNetV2 (frozen) → GlobalAvgPool2D → Dense(128, ReLU) → Dropout(0.3) → Dense(9, Softmax) |
| Callbacks | ModelCheckpoint (best `val_accuracy`) + EarlyStopping (patience = 5) |
| Best val accuracy | **85.37%** (Epoch 11) |
| Final test accuracy | **87%** (weighted F1 = 0.86) |

> ⚠️ **Note:** Classes with very low test support (`sour`: 3 samples, `immature`: 4, `husk`: 5) show degraded F1 scores. Additional data collection is the highest-priority improvement for this model.

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
- Anthropic API key (for plain-language explanations)

---

### Step 1 — Set up Python environment

```bash
conda create -n coffee-ml python=3.10
conda activate coffee-ml
conda install pandas numpy matplotlib seaborn scikit-learn jupyter
pip install xgboost shap fastapi uvicorn kagglehub joblib tensorflow roboflow
```

### Step 2 — Train models (Jupyter Notebooks)

Run the three notebooks in order from the repo root:

```bash
jupyter notebook
```

| Notebook | What it does | Output |
|----------|-------------|--------|
| `notebooks/coffee_quality_model.ipynb` | EDA → Linear Regression → XGBoost (GridSearchCV) → K-Means → SHAP | `.pkl` files in `outputs/models/` |
| `notebooks/roast_classifier.ipynb` | MobileNetV2 transfer learning on bean imagery | `.h5` file in `outputs/models/` |
| `notebooks/defect_detector.ipynb` | Roboflow API integration test for defect detection | Requires Roboflow API key |

All exported model files are saved to `outputs/models/`.

### Step 3 — Start the FastAPI ML microservice

```bash
cd ml_service
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001
```

Verify it's running: `curl http://localhost:8001/health`

### Step 4 — Start the Go backend

```bash
cd backend
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY and other settings
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
| `POST` | `/api/compare` | Side-by-side comparison of two coffee samples |
| `POST` | `/api/predict-roast` | Classify roast level from uploaded bean image |
| `POST` | `/api/predict-defect` | Detect and classify defects from uploaded bean image |
| `GET`  | `/api/regions` | Ethiopian regional quality data and rankings |
| `GET`  | `/api/health` | Backend + ML service health check |

### FastAPI ML Service — port 8001

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict` | XGBoost inference + SHAP values + K-Means cluster assignment |
| `POST` | `/predict-roast` | CNN roast level classification |
| `POST` | `/predict-defect` | Roboflow-hosted defect detection |
| `GET`  | `/regions` | Precomputed regional quality statistics |
| `GET`  | `/health` | Model loading status for all models |

#### Example — Quality Prediction Request

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "aroma": 8.17,
    "flavor": 8.19,
    "aftertaste": 8.08,
    "acidity": 8.07,
    "body": 7.98,
    "balance": 8.05,
    "altitude": 2000,
    "moisture": 11.5,
    "processing_method": "Washed / Wet"
  }'
```

---

## 7. Frontend Pages

| Route | Description |
|-------|-------------|
| `/` | Landing page — hero, how-it-works flow, feature grid |
| `/predict` | Quality predictor form with real-time SHAP explanation output |
| `/predict/roast` | Upload a bean photo → roast level prediction with confidence |
| `/predict/defect` | Upload a bean photo → defect bounding boxes overlaid on image |
| `/compare` | Side-by-side comparison of two coffee samples across all dimensions |
| `/explore` | Ethiopian regional quality rankings, maps, and score distributions |

---

## 8. Environment Variables

### Backend — `backend/.env`

```env
PYTHON_SERVICE_URL=http://localhost:8001
ANTHROPIC_API_KEY=your_anthropic_api_key_here
PORT=8000
```

### Frontend — `frontend/.env.local`

```env
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

---

## 9. Results & Performance

### Model Summary

| Model | Type | Key Metric | Score | Status |
|-------|------|-----------|-------|--------|
| Quality Scorer (Linear Regression) | Regression | Test R² | 0.994 | ✅ Production Ready |
| Quality Scorer (XGBoost) | Regression | Test R² | 0.960 | ⚠️ Needs more data |
| Roast Classifier | CNN — 4 class | Test Accuracy | 99.5% | ✅ Production Ready |
| Defect Detector | CNN — 9 class | Test Accuracy | 87% | ⚠️ Improve minority classes |
| Flavor Clustering | Unsupervised | Clusters (k) | 4 | ✅ Deployed |

### Ethiopian Regional Quality

All tested Ethiopian regions exceed the SCA Specialty Grade threshold of 80 points:

| Region | Avg Predicted Score | Notes |
|--------|-------------------|-------|
| Guji | ~86.0 | Highest — complex fruit-forward profiles |
| OROMIA | ~84.5 | Consistently above specialty grade |
| Sidama | ~84.0 | Reliable specialty grade |

---

## 10. Tech Stack

| Layer | Technologies |
|-------|-------------|
| **ML** | scikit-learn, XGBoost, SHAP, K-Means, TensorFlow / Keras, MobileNetV2 |
| **ML Service** | FastAPI, Uvicorn, joblib, kagglehub, Roboflow |
| **Backend** | Go, Gin framework |
| **Frontend** | Next.js 14, React 18, Tailwind CSS, Lucide icons |
| **Data Sources** | Kaggle (CQI cupping data, coffee bean images), Roboflow Universe (defect images) |
| **Explainability** | SHAP TreeExplainer + Anthropic API (plain-language summaries) |

---

## 11. Build Order

| Week | Focus | Deliverables |
|------|-------|-------------|
| **Week 1** | Jupyter notebooks | Data loading → EDA → model training → export `.pkl` / `.h5` / `.json` files |
| **Week 2** | Backend services | FastAPI microservice + Go backend + integration tests |
| **Week 3** | Frontend + polish | Next.js 6-page app + styling + final report |

---

## 12. Deliverables

- [ ] `notebooks/coffee_quality_model.ipynb` — XGBoost + K-Means + SHAP
- [ ] `notebooks/roast_classifier.ipynb` — MobileNetV2 roast classification
- [ ] `notebooks/defect_detector.ipynb` — Roboflow defect detection
- [ ] `outputs/models/` — all exported model files (`.pkl`, `.h5`, `.json`)
- [ ] `ml_service/` — FastAPI microservice
- [ ] `backend/` — Go HTTP server
- [ ] `frontend/` — Next.js web application (6 pages)
- [ ] Written technical report with EDA findings, model comparison, SHAP analysis, and Ethiopian regional insights

---

## 13. References

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