from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import binascii, joblib, json, os, base64, logging
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import numpy as np
import pandas as pd

# TensorFlow and SHAP are imported lazily inside load_models() so the API layer
# can be imported and tested without the heavy ML runtime installed.

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="Coffee Quality ML Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

xgb_model = None
shap_explainer = None
kmeans_model = None
scaler = None
cluster_labels = None
feature_cols = None
roast_model = None
roast_classes = None
regions_data = None
defect_model = None
defect_classes = None

SENSORY_COLS = ['Aroma', 'Flavor', 'Acidity', 'Body', 'Balance']
SPECIALTY_THRESHOLD = 80


class PredictRequest(BaseModel):
    features: dict

class PredictRoastRequest(BaseModel):
    image_base64: str

class PredictDefectRequest(BaseModel):
    image_base64: str


@app.on_event("startup")
def load_models():
    global xgb_model, shap_explainer, kmeans_model, scaler
    global cluster_labels, feature_cols, roast_model, roast_classes, regions_data
    global defect_model, defect_classes

    import shap  # noqa: F401  (needed to unpickle the TreeExplainer)
    import tensorflow as tf

    xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl"))
    shap_explainer = joblib.load(os.path.join(MODELS_DIR, "shap_explainer.pkl"))
    kmeans_model = joblib.load(os.path.join(MODELS_DIR, "kmeans_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

    with open(os.path.join(MODELS_DIR, "cluster_labels.json")) as f:
        cluster_labels = json.load(f)
    with open(os.path.join(MODELS_DIR, "feature_columns.json")) as f:
        feature_cols = json.load(f)
    with open(os.path.join(MODELS_DIR, "regions.json")) as f:
        regions_data = json.load(f)

    roast_model = tf.keras.models.load_model(
        os.path.join(MODELS_DIR, "roast_classifier.h5")
    )
    with open(os.path.join(MODELS_DIR, "roast_classes.json")) as f:
        roast_classes = json.load(f)

    defect_model = tf.keras.models.load_model(
        os.path.join(MODELS_DIR, "defect_detector.h5")
    )
    with open(os.path.join(MODELS_DIR, "defect_classes.json")) as f:
        defect_classes = json.load(f)

    logger.info(f"roast_model input shape: {roast_model.input_shape}")
    logger.info(f"defect_model input shape: {defect_model.input_shape}")
    logger.info(f"roast_classes: {roast_classes}")
    logger.info(f"defect_classes: {defect_classes}")
    logger.info("All models loaded successfully.")


def idx_to_class_name(classes, idx):
    """Handles both list format (['black','broken',...]) and dict format ({'black':0,...})."""
    if isinstance(classes, list):
        return classes[idx]
    if isinstance(classes, dict):
        # could be {name: index} or {index: name}
        inverted = {v: k for k, v in classes.items()}
        if idx in inverted:
            return inverted[idx]
        if str(idx) in classes:
            return classes[str(idx)]
    raise ValueError(f"Could not resolve class name for index {idx} in {classes}")


def get_target_size(model, fallback=(224, 224)):
    """Reads the model's actual expected input size instead of hardcoding it."""
    try:
        shape = model.input_shape  # e.g. (None, 224, 224, 3)
        h, w = shape[1], shape[2]
        if h and w:
            return (w, h)  # PIL expects (width, height)
    except Exception:
        pass
    return fallback


def decode_image(image_base64: str, model) -> np.ndarray:
    """Base64 -> normalised (1, H, W, 3) float32 batch. Raises 400 on bad input."""
    try:
        img_data = base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="image_base64 is not valid base64")
    try:
        img = Image.open(BytesIO(img_data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="image_base64 does not decode to a supported image")
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(get_target_size(model))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def require_loaded(*objects):
    """503 (not 500) while models are still loading or failed to load."""
    if any(o is None for o in objects):
        raise HTTPException(status_code=503, detail="Models are not loaded")


def classify_image(model, classes, image_base64: str):
    """Shared roast/defect path: returns (class_name, confidence_percent)."""
    require_loaded(model, classes)
    arr = decode_image(image_base64, model)
    preds = model(arr, training=False).numpy()
    idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][idx])
    return idx_to_class_name(classes, idx), round(confidence * 100, 1)


@app.get("/")
def root():
    return {"status": "ok", "service": "coffee-quality-ml-service"}


@app.get("/health")
def health():
    loaded = [
        xgb_model, shap_explainer, kmeans_model, scaler,
        cluster_labels, feature_cols, roast_model, roast_classes,
        defect_model, defect_classes,
    ]
    return {"status": "ok", "modelsLoaded": all(x is not None for x in loaded)}


@app.post("/predict")
def predict(req: PredictRequest):
    require_loaded(xgb_model, shap_explainer, kmeans_model, scaler, cluster_labels, feature_cols)
    missing = [c for c in SENSORY_COLS if c not in req.features]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required sensory features: {missing}")
    try:
        input_df = pd.DataFrame([req.features])
        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_cols]
        X = scaler.transform(input_df.values)

        score = float(xgb_model.predict(X)[0])
        shap_vals = shap_explainer.shap_values(X)[0]
        top_idx = np.argsort(np.abs(shap_vals))[::-1][:4]
        top_features = [
            {
                "feature": feature_cols[i],
                "value": float(shap_vals[i]),
                "direction": "positive" if shap_vals[i] > 0 else "negative"
            }
            for i in top_idx
        ]

        sensory_input = pd.DataFrame([req.features])[SENSORY_COLS].values
        cluster_id = int(kmeans_model.predict(sensory_input)[0])
        flavor_cluster = {
            "id": cluster_id,
            "name": cluster_labels[str(cluster_id)]["name"],
            "description": cluster_labels[str(cluster_id)]["description"]
        }

        grade = "Specialty" if score >= SPECIALTY_THRESHOLD else "Below Specialty"
        counterfactual = None
        if score < SPECIALTY_THRESHOLD:
            counterfactual = {
                "suggestion": "Consider improving processing method or growing conditions."
            }

        return {
            "score": round(score, 1),
            "grade": grade,
            "shap": top_features,
            "flavorCluster": flavor_cluster,
            "counterfactual": counterfactual
        }
    except Exception as e:
        logger.exception(f"/predict failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-roast")
def predict_roast(req: PredictRoastRequest):
    class_name, confidence = classify_image(roast_model, roast_classes, req.image_base64)
    return {
        "roastLevel": str(class_name).lower(),
        "confidence": confidence
    }


@app.post("/predict-defect")
def predict_defect(req: PredictDefectRequest):
    class_name, confidence = classify_image(defect_model, defect_classes, req.image_base64)
    # NOTE: the training data has no "good"/no-defect class, so this flag is
    # currently always True. See README "Limitations & Evaluation Notes".
    is_defective = str(class_name).lower() != "good"

    return {
        "defects": [{
            "class": class_name,
            "confidence": confidence,
        }],
        "isDefective": is_defective
    }


@app.get("/regions")
def get_regions():
    return regions_data