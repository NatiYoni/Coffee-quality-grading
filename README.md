# Ethiopian Coffee Quality Predictor

A machine learning system that predicts coffee quality scores from cupping/origin data, explains predictions in plain language, and visualizes quality patterns across Ethiopian coffee-growing regions.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Next.js       │────▶│   Go Backend    │────▶│   FastAPI       │
│   Frontend      │     │   (port 8000)   │     │   (port 8001)   │
│   (6 pages)     │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                               ┌────────┴────────┐
                                               │  ML Models      │
                                               │  - XGBoost      │
                                               │  - K-means      │
                                               │  - CNN (roast)  │
                                               │  - SHAP         │
                                               └─────────────────┘
```

## Project Structure

```
coffee-quality-predictor/
├── notebooks/
│   ├── coffee_quality_model.ipynb   # XGBoost + K-means + SHAP
│   ├── roast_classifier.ipynb       # MobileNetV2 transfer learning
│   └── defect_detector.ipynb        # Roboflow API test
├── outputs/
│   ├── charts/                      # EDA + training PNGs
│   └── models/                      # Exported .pkl/.h5/.json files
├── ml_service/
│   ├── main.py                      # FastAPI app (port 8001)
│   └── requirements.txt
├── backend/
│   ├── main.go                      # Go HTTP server (port 8000)
│   ├── go.mod
│   └── .env.example
├── frontend/
│   ├── app/
│   │   ├── layout.js
│   │   ├── page.js                  # Landing page
│   │   ├── predict/page.js          # Quality predictor form
│   │   ├── predict/roast/page.js    # Roast level classifier
│   │   ├── predict/defect/page.js   # Defect detector
│   │   ├── compare/page.js          # Side-by-side comparison
│   │   └── explore/page.js          # Ethiopian regions
│   ├── package.json
│   ├── next.config.js
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   └── app/globals.css
└── README.md
```

## Quick Start

### 1. Set up Python environment (Anaconda)

```bash
conda create -n coffee-ml python=3.10
conda activate coffee-ml
conda install pandas numpy matplotlib seaborn scikit-learn jupyter
pip install xgboost shap fastapi uvicorn kagglehub joblib tensorflow roboflow
```

### 2. Train models (Jupyter Notebooks)

Open Jupyter and run notebooks in order:

```bash
jupyter notebook
```

1. `notebooks/coffee_quality_model.ipynb` — trains XGBoost, K-means, exports `.pkl` files
2. `notebooks/roast_classifier.ipynb` — trains MobileNetV2, exports `.h5` file
3. `notebooks/defect_detector.ipynb` — tests Roboflow API (requires API key)

All exported models go to `outputs/models/`.

### 3. Start FastAPI microservice

```bash
cd ml_service
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001
```

### 4. Start Go backend

```bash
cd backend
cp .env.example .env
# Edit .env with your settings
go mod tidy
go run main.go
```

### 5. Start Next.js frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000

## API Endpoints

### Go Backend (port 8000)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/predict` | Predict quality score from features |
| POST | `/api/compare` | Compare two coffee samples |
| POST | `/api/predict-roast` | Classify roast level from image |
| POST | `/api/predict-defect` | Detect defects from image |
| GET | `/api/regions` | Get Ethiopian region quality data |
| GET | `/api/health` | Health check |

### FastAPI (port 8001)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/predict` | XGBoost + SHAP + K-means prediction |
| POST | `/predict-roast` | CNN roast classification |
| POST | `/predict-defect` | Roboflow defect detection |
| GET | `/regions` | Regional quality data |
| GET | `/health` | Model loading status |

## Environment Variables

### Backend (.env)

```
PYTHON_SERVICE_URL=http://localhost:8001
ANTHROPIC_API_KEY=your_key_here
PORT=8000
```

### Frontend (.env.local)

```
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

## Model Details

### Quality Model (XGBoost)
- **Target**: Total.Cup.Points (0-100)
- **Features**: Altitude, Aroma, Flavor, Aftertaste, Acidity, Body, Balance, Uniformity, Clean.Cup, Sweetness, Moisture, Defects, Processing.Method, Country
- **Baseline**: Linear Regression
- **Best Model**: XGBoost with GridSearchCV (max_depth, n_estimators, learning_rate)
- **Explainability**: SHAP TreeExplainer

### Flavor Clustering (K-means)
- **Features**: Aroma, Flavor, Acidity, Body, Balance
- **Clusters**: 4 (Bright & Floral, Earthy & Full-bodied, Balanced & Sweet, Mild & Neutral)

### Roast Classifier (MobileNetV2)
- **Classes**: green, light, medium, dark
- **Architecture**: Transfer learning with frozen ImageNet backbone

### Defect Detector (Roboflow)
- **Model**: coffee-bean-defect-smvw1 (hosted)
- **Classes**: Various bean defect types

## Pages

| Route | Description |
|-------|-------------|
| `/` | Landing page with hero, how-it-works, features grid |
| `/predict` | Quality predictor form with SHAP results |
| `/predict/roast` | Upload bean photo → roast level |
| `/predict/defect` | Upload bean photo → defect bounding boxes |
| `/compare` | Side-by-side two-coffee comparison |
| `/explore` | Ethiopian region quality rankings |

## Tech Stack

- **ML**: scikit-learn, XGBoost, SHAP, K-means, TensorFlow/Keras
- **Backend**: FastAPI (Python), Gin (Go)
- **Frontend**: Next.js 14, React 18, Tailwind CSS, Lucide icons
- **Data**: Kaggle (CQI coffee quality, coffee bean images), Roboflow (defects)

## Build Order

- **Week 1**: Jupyter notebooks (data loading → EDA → training → export)
- **Week 2**: FastAPI microservice + Go backend + integration tests
- **Week 3**: Next.js frontend + polish + report

## Deliverables

1. Three Jupyter Notebooks
2. Exported model files in `outputs/models/`
3. FastAPI microservice
4. Go backend
5. Next.js frontend (6 pages)
6. Written report with EDA findings, model comparison, SHAP analysis, regional insights