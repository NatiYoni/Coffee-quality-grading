import base64

import numpy as np
import pytest

import main
from conftest import SENSORY, FEATURE_COLS, png_base64


# --- health ---------------------------------------------------------------

def test_health_reports_not_loaded_before_startup(client, unloaded):
    body = client.get("/health").json()
    assert body == {"status": "ok", "modelsLoaded": False}


def test_health_reports_loaded_when_every_artifact_is_present(client, loaded):
    assert client.get("/health").json()["modelsLoaded"] is True


def test_health_is_false_if_a_single_artifact_is_missing(client, loaded, monkeypatch):
    monkeypatch.setattr(main, "defect_classes", None)
    assert client.get("/health").json()["modelsLoaded"] is False


# --- /predict -------------------------------------------------------------

def test_predict_returns_full_response_shape(client, loaded):
    r = client.post("/predict", json={"features": {**SENSORY, "Altitude": 1900}})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"score", "grade", "shap", "flavorCluster", "counterfactual"}
    assert body["score"] == 84.2
    assert body["grade"] == "Specialty"
    assert body["counterfactual"] is None
    assert body["flavorCluster"] == {
        "id": 2, "name": "Balanced & Sweet", "description": "High balance and sweetness",
    }


def test_predict_shap_lists_top_four_by_magnitude_with_direction(client, loaded):
    body = client.post("/predict", json={"features": SENSORY}).json()
    assert [f["feature"] for f in body["shap"]] == ["Flavor", "Balance", "Altitude", "Aroma"]
    assert body["shap"][0]["direction"] == "positive"
    assert body["shap"][2] == {"feature": "Altitude", "value": -0.3, "direction": "negative"}


def test_predict_fills_missing_features_with_zero_in_training_column_order(client, loaded):
    client.post("/predict", json={"features": {**SENSORY, "Altitude": 1500}})
    seen = loaded["xgb_model"].seen
    assert seen.shape == (1, len(FEATURE_COLS))
    row = dict(zip(FEATURE_COLS, seen[0]))
    assert row["Altitude"] == 1500
    assert row["Aroma"] == 8.2
    assert row["Uniformity"] == 0 and row["Category Two Defects"] == 0


def test_predict_uses_only_sensory_columns_for_clustering(client, loaded):
    client.post("/predict", json={"features": {**SENSORY, "Altitude": 1500}})
    np.testing.assert_allclose(loaded["kmeans_model"].seen, [[8.2, 8.1, 8.0, 7.9, 8.0]])


@pytest.mark.parametrize("score,grade", [(79.9, "Below Specialty"), (80.0, "Specialty"), (80.1, "Specialty")])
def test_predict_grade_threshold_is_80_points(client, loaded, score, grade):
    loaded["xgb_model"].score = score
    body = client.post("/predict", json={"features": SENSORY}).json()
    assert body["grade"] == grade
    assert (body["counterfactual"] is not None) == (grade == "Below Specialty")


def test_predict_rejects_missing_sensory_features_with_422(client, loaded):
    r = client.post("/predict", json={"features": {"Aroma": 8.0, "Flavor": 8.0, "Altitude": 1800}})
    assert r.status_code == 422
    assert "Acidity" in r.json()["detail"] and "Body" in r.json()["detail"]


def test_predict_rejects_malformed_body_with_422(client, loaded):
    assert client.post("/predict", json={"feature": SENSORY}).status_code == 422


def test_predict_returns_503_while_models_are_not_loaded(client, unloaded):
    r = client.post("/predict", json={"features": SENSORY})
    assert r.status_code == 503


# --- image endpoints ------------------------------------------------------

def test_predict_roast_returns_lowercased_class_and_percent_confidence(client, loaded):
    r = client.post("/predict-roast", json={"image_base64": png_base64()})
    assert r.status_code == 200
    assert r.json() == {"roastLevel": "light", "confidence": 90.0}


def test_predict_roast_resizes_to_model_input_and_normalises_to_unit_range(client, loaded):
    model = loaded["roast_model"]
    model.input_shape = (None, 96, 128, 3)  # (batch, height, width, channels)
    client.post("/predict-roast", json={"image_base64": png_base64(size=(300, 40))})
    assert model.seen.shape == (1, 96, 128, 3)
    assert model.seen.dtype == np.float32
    assert 0.0 <= model.seen.min() and model.seen.max() <= 1.0


def test_predict_roast_converts_grayscale_to_rgb(client, loaded):
    r = client.post("/predict-roast", json={"image_base64": png_base64(mode="L")})
    assert r.status_code == 200
    assert loaded["roast_model"].seen.shape[-1] == 3


def test_predict_defect_resolves_name_from_dict_class_mapping(client, loaded):
    r = client.post("/predict-defect", json={"image_base64": png_base64()})
    assert r.status_code == 200
    assert r.json() == {"defects": [{"class": "broken", "confidence": 70.0}], "isDefective": True}


@pytest.mark.parametrize("endpoint", ["/predict-roast", "/predict-defect"])
def test_invalid_base64_is_a_client_error_not_a_500(client, loaded, endpoint):
    r = client.post(endpoint, json={"image_base64": "not base64!!"})
    assert r.status_code == 400
    assert "base64" in r.json()["detail"]


@pytest.mark.parametrize("endpoint", ["/predict-roast", "/predict-defect"])
def test_non_image_bytes_are_a_client_error_not_a_500(client, loaded, endpoint):
    payload = base64.b64encode(b"definitely not an image").decode()
    r = client.post(endpoint, json={"image_base64": payload})
    assert r.status_code == 400
    assert "image" in r.json()["detail"]


def test_image_endpoints_return_503_while_models_are_not_loaded(client, unloaded):
    r = client.post("/predict-roast", json={"image_base64": png_base64()})
    assert r.status_code == 503


# --- helpers & misc -------------------------------------------------------

def test_idx_to_class_name_supports_list_and_both_dict_orientations():
    assert main.idx_to_class_name(["Dark", "Light"], 1) == "Light"
    assert main.idx_to_class_name({"Dark": 0, "Light": 1}, 1) == "Light"
    assert main.idx_to_class_name({"0": "Dark", "1": "Light"}, 1) == "Light"
    with pytest.raises(ValueError):
        main.idx_to_class_name({"Dark": 0}, 5)


def test_get_target_size_reads_model_shape_and_falls_back():
    class M:
        input_shape = (None, 96, 128, 3)

    assert main.get_target_size(M()) == (128, 96)
    assert main.get_target_size(object()) == (224, 224)


def test_regions_returns_precomputed_data(client, loaded):
    assert client.get("/regions").json() == [{"region": "Guji", "avg_score": 86.0}]


def test_root_identifies_the_service(client):
    assert client.get("/").json()["service"] == "coffee-quality-ml-service"
