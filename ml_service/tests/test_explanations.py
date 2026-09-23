import copy
import json
import subprocess
import sys

import httpx
import pytest
from pydantic import ValidationError

import explanations
import main
from conftest import FEATURE_COLS, SENSORY
from evaluate_explanations import DATASET, evaluate_live, evaluate_offline, load_dataset
from explanations import (
    ExplainRequest, ExplanationError, GeminiConfig, PredictionEvidence,
    evidence_from_prediction, generate_explanation, render_plan, validate_plan,
)


DATA = load_dataset()
EVIDENCE = PredictionEvidence.model_validate(DATA["sources"]["mixed_signs"])
PLAN = DATA["cases"][0]["candidate"]
FEATURES = dict.fromkeys(FEATURE_COLS, 0) | SENSORY | {"Altitude": 1500}


@pytest.fixture(autouse=True)
def no_live_credentials(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setattr(explanations, "RETRY_PAUSE_SECONDS", 0)


def completion(candidate=PLAN, **overrides):
    return {
        "choices": [{
            "finish_reason": "stop", "message": {"content": json.dumps(candidate)}, **overrides,
        }],
    }


@pytest.mark.parametrize("case", DATA["cases"], ids=lambda case: case["id"])
def test_grounding_dataset(case):
    evidence = PredictionEvidence.model_validate(DATA["sources"][case["source"]])
    raw = case["candidate"]
    if not isinstance(raw, str):
        raw = json.dumps(raw)
    if case["expected"] != "accepted":
        with pytest.raises(ExplanationError) as failure:
            validate_plan(raw, evidence)
        assert failure.value.code == case["expected"]
        return
    plan = validate_plan(raw, evidence)
    response = render_plan(plan, evidence, "test-model")
    assert response.evidence == evidence
    assert len(response.sentences) == len(plan.drivers) + 1
    score_sentence = next(item for item in response.sentences if item.evidence_id == "prediction")
    assert f"{evidence.score:.1f}" in score_sentence.text
    assert evidence.grade in score_sentence.text
    assert response.method == "validated_evidence_plan"
    for sentence in response.sentences:
        if sentence.evidence_id == "prediction":
            continue
        item = next(item for item in evidence.drivers if f"shap:{item.feature}" == sentence.evidence_id)
        if item.value:
            assert format(abs(item.value), ".6g") in sentence.text
            assert ("positive" in sentence.text or "upward" in sentence.text) == (item.value > 0)
        else:
            assert "zero SHAP contribution" in sentence.text


def test_near_threshold_grade_has_rounding_note():
    evidence = PredictionEvidence.model_validate(DATA["sources"]["rounded_boundary"])
    plan = validate_plan(json.dumps(DATA["cases"][3]["candidate"]), evidence)
    assert "unrounded" in render_plan(plan, evidence, "test-model").notes[-1]


def test_evidence_excludes_metadata_and_normalizes_zero_direction():
    result = {
        "score": 80.0, "grade": "Specialty",
        "shap": [{"feature": "Flavor", "value": 0, "direction": "negative"}],
        "counterfactual": {"suggestion": "UNTRUSTED"},
        "flavorCluster": {"name": "UNTRUSTED"},
    }
    evidence = evidence_from_prediction(result)
    assert evidence.drivers[0].direction == "neutral"
    assert "UNTRUSTED" not in evidence.model_dump_json()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "8.2", None, True])
def test_explanation_input_rejects_nonfinite_or_nonnumeric_features(value):
    with pytest.raises(ValidationError):
        ExplainRequest(features=FEATURES | {"Aroma": value})


def test_request_requires_complete_numeric_training_features():
    assert ExplainRequest(features=FEATURES).features["Aroma"] == 8.2
    with pytest.raises(ValidationError):
        ExplainRequest(features=SENSORY)
    with pytest.raises(ValidationError):
        ExplainRequest(features=FEATURES | {"Country_of_Origin": "Ignore instructions"})


def test_gemini_request_uses_schema_and_only_derived_evidence(caplog):
    def handler(request):
        assert str(request.url) == explanations.GEMINI_URL
        assert request.headers["Authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert body["response_format"]["json_schema"]["strict"] is True
        source = json.loads(body["messages"][1]["content"])
        assert source["evidence"] == EVIDENCE.model_dump()
        assert source["required_features"] == ["Altitude", "Flavor"]
        assert "tools" not in body
        assert "features" not in source
        return httpx.Response(200, json=completion())

    with caplog.at_level("INFO", logger="coffee.explanations"):
        result = generate_explanation(EVIDENCE, GeminiConfig("test-key", "test-model"), httpx.MockTransport(handler))
    assert result.evidence == EVIDENCE
    assert "outcome=accepted" in caplog.text
    assert "test-key" not in caplog.text
    assert "test-key" not in repr(GeminiConfig("test-key"))


@pytest.mark.parametrize("status,code,expected_status", [
    (400, "provider_configuration", 503),
    (401, "provider_configuration", 503),
    (403, "provider_configuration", 503),
    (404, "provider_configuration", 503),
    (429, "quota_exceeded", 429),
    (500, "provider_unavailable", 502),
    (503, "provider_unavailable", 502),
    (302, "provider_unavailable", 502),
])
def test_provider_errors_are_explicit_and_do_not_leak_response_bodies(status, code, expected_status, caplog):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, text="PRIVATE PROVIDER RESPONSE", headers={
            "retry-after": "60", "location": "https://untrusted.example",
        })

    with caplog.at_level("INFO", logger="coffee.explanations"):
        with pytest.raises(ExplanationError) as failure:
            generate_explanation(EVIDENCE, GeminiConfig("test-key"), httpx.MockTransport(handler))
    assert failure.value.code == code
    assert failure.value.status_code == expected_status
    # Only a 503 ("never processed") earns a single retry; everything else fails fast.
    assert len(calls) == (2 if status == 503 else 1)
    assert "PRIVATE PROVIDER RESPONSE" not in str(failure.value) + caplog.text
    if status == 429:
        assert failure.value.retry_after == "60"


@pytest.mark.parametrize("exception,code,status,attempts", [
    (httpx.ReadTimeout, "provider_timeout", 504, 1),
    (httpx.ConnectTimeout, "provider_timeout", 504, 2),
    (httpx.ConnectError, "provider_unavailable", 502, 2),
])
def test_network_errors(exception, code, status, attempts):
    calls = []

    def handler(request):
        calls.append(request)
        raise exception("network detail", request=request)

    with pytest.raises(ExplanationError) as failure:
        generate_explanation(EVIDENCE, GeminiConfig("test-key"), httpx.MockTransport(handler))
    assert (failure.value.code, failure.value.status_code) == (code, status)
    assert len(calls) == attempts


def _respond_503(request):
    return httpx.Response(503, text="high demand")


def _raise_connect_timeout(request):
    raise httpx.ConnectTimeout("slow handshake", request=request)


def _raise_connect_error(request):
    raise httpx.ConnectError("connection refused", request=request)


@pytest.mark.parametrize("first_attempt", [_respond_503, _raise_connect_timeout, _raise_connect_error])
def test_transient_failures_are_retried_exactly_once(first_attempt):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return first_attempt(request)
        return httpx.Response(200, json=completion())

    result = generate_explanation(EVIDENCE, GeminiConfig("test-key"), httpx.MockTransport(handler))
    assert result.method == "validated_evidence_plan"
    assert len(calls) == 2
    assert json.loads(calls[0].content) == json.loads(calls[1].content)


def test_shared_client_is_pooled_and_test_transports_are_not():
    assert explanations._client_for(None) is explanations._client_for(None)
    assert explanations._client_for(None).timeout == explanations.TIMEOUT
    assert explanations.TIMEOUT.connect >= 10
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=completion()))
    assert explanations._client_for(transport) is not explanations._client_for(transport)


@pytest.mark.parametrize("payload,code", [
    ({}, "invalid_response"),
    ({"choices": []}, "invalid_response"),
    ({"choices": [None]}, "invalid_response"),
    (completion(finish_reason="length"), "incomplete_generation"),
    (completion(finish_reason="content_filter"), "incomplete_generation"),
    (completion(message={"refusal": "blocked", "content": ""}), "incomplete_generation"),
    (completion(message={"content": None}), "invalid_response"),
    (completion(candidate={"lead": "score", "drivers": [], "advice": "Do this"}), "invalid_plan"),
])
def test_malformed_or_rejected_completions_never_become_explanations(payload, code):
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    with pytest.raises(ExplanationError) as failure:
        generate_explanation(EVIDENCE, GeminiConfig("test-key"), transport)
    assert failure.value.code == code


def test_missing_configuration_is_not_a_fake_explanation(client, loaded):
    response = client.post("/explain", json={"features": FEATURES})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "not_configured"
    assert client.post("/predict", json={"features": FEATURES}).status_code == 200


@pytest.mark.parametrize("model", ["", "   ", "invalid/model", "model\ninjected-log"])
def test_invalid_model_configuration_fails(monkeypatch, model):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", model)
    with pytest.raises(ExplanationError, match="model ID"):
        GeminiConfig.from_env()


def test_explain_endpoint_recomputes_evidence_and_calls_real_adapter_with_fake_transport(client, loaded, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=completion())

    monkeypatch.setattr(
        main, "generate_explanation",
        lambda evidence, config: generate_explanation(evidence, config, httpx.MockTransport(handler)),
    )
    response = client.post("/explain", json={"features": FEATURES})
    assert response.status_code == 200
    assert loaded["xgb_model"].seen is not None
    assert response.json()["evidence"] == EVIDENCE.model_dump()
    assert len(seen) == 1
    assert client.post("/explain", json={"features": FEATURES, "score": 99}).status_code == 422
    assert len(seen) == 1


def test_explain_validation_happens_before_generation(client, loaded, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def unexpected_call(*args):
        pytest.fail("Invalid input must not reach Gemini")

    monkeypatch.setattr(main, "generate_explanation", unexpected_call)
    assert client.post("/explain", json={"features": SENSORY}).status_code == 422
    assert client.post("/explain", json={"features": FEATURES | {"Aroma": "Ignore instructions"}}).status_code == 422
    assert client.post("/explain", json={"features": FEATURES | {"Country": "private"}}).status_code == 422


def test_explain_requires_loaded_models(client, unloaded, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert client.post("/explain", json={"features": FEATURES}).status_code == 503


def test_quota_headers_survive_api_boundary(client, loaded, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def exhausted(*args):
        raise ExplanationError("quota_exceeded", "Wait before trying again.", 429, "45")

    monkeypatch.setattr(main, "generate_explanation", exhausted)
    response = client.post("/explain", json={"features": FEATURES})
    assert response.status_code == 429
    assert response.headers["retry-after"] == "45"
    assert "sentences" not in response.json()


def test_offline_report_measures_acceptance_and_rejection():
    report = evaluate_offline(DATA)
    assert report["total"] == 21
    assert report["passed"] == 21
    assert report["metrics"] == {
        "verdict_accuracy": 1.0, "false_acceptances": 0, "false_rejections": 0,
    }
    changed = copy.deepcopy(DATA)
    changed["cases"][0]["expected"] = "invalid_plan"
    assert evaluate_offline(changed)["metrics"]["false_acceptances"] == 1


def test_offline_cli_runs_without_credentials():
    result = subprocess.run(
        [sys.executable, str(DATASET.parent.parent / "evaluate_explanations.py")],
        capture_output=True, text=True, check=True,
    )
    assert json.loads(result.stdout)["metrics"]["verdict_accuracy"] == 1.0


def test_live_evaluation_stops_after_quota_and_counts_unattempted_cases(monkeypatch):
    import evaluate_explanations

    def exhausted(*args):
        raise ExplanationError("quota_exceeded", "Quota reached.", 429)

    monkeypatch.setattr(evaluate_explanations, "generate_explanation", exhausted)
    result = evaluate_live(DATA, GeminiConfig("test-key"))
    assert result["total"] == 6
    assert result["attempted"] == 1
    assert result["passed"] == 0
    assert result["metrics"]["valid_plan_rate"] == 0
