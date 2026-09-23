"""Offline guardrail evaluation by default; --live explicitly calls Gemini."""

import argparse
import json
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv

from explanations import (
    ExplanationError, GeminiConfig, PredictionEvidence, generate_explanation,
    render_plan, validate_plan,
)


DATASET = Path(__file__).parent / "evals" / "explanation_cases.json"


def load_dataset():
    data = json.loads(DATASET.read_text())
    if data["data_kind"] != "synthetic" or not data["sources"] or not data["cases"]:
        raise ValueError("The evaluator requires a nonempty synthetic dataset")
    return data


def evaluate_offline(data):
    cases = []
    for case in data["cases"]:
        evidence = PredictionEvidence.model_validate(data["sources"][case["source"]])
        raw = case["candidate"]
        if not isinstance(raw, str):
            raw = json.dumps(raw)
        try:
            plan = validate_plan(raw, evidence)
            render_plan(plan, evidence, "offline-fixture")
            actual = "accepted"
        except ExplanationError as exc:
            actual = exc.code
        cases.append({
            "id": case["id"], "expected": case["expected"], "actual": actual,
            "passed": actual == case["expected"],
        })
    invalid = [case for case in cases if case["expected"] != "accepted"]
    valid = [case for case in cases if case["expected"] == "accepted"]
    if not invalid or not valid:
        raise ValueError("Evaluation must contain both acceptable and unacceptable plans")
    passed = sum(case["passed"] for case in cases)
    return {
        "mode": "offline_guardrails",
        "dataset_version": data["version"],
        "total": len(cases),
        "passed": passed,
        "metrics": {
            "verdict_accuracy": passed / len(cases),
            "false_acceptances": sum(case["actual"] == "accepted" for case in invalid),
            "false_rejections": sum(case["actual"] != "accepted" for case in valid),
        },
        "cases": cases,
    }


def evaluate_live(data, config, source_id=None):
    cases = []
    sources = data["sources"] if source_id is None else {source_id: data["sources"][source_id]}
    for name, raw in sources.items():
        evidence = PredictionEvidence.model_validate(raw)
        started = time.monotonic()
        try:
            explanation = generate_explanation(evidence, config)
            case = {"id": name, "passed": True, "explanation": explanation.model_dump()}
        except ExplanationError as exc:
            case = {"id": name, "passed": False, "error": exc.code}
        case["latency_ms"] = round((time.monotonic() - started) * 1000)
        cases.append(case)
        if not case["passed"] and case.get("error") == "quota_exceeded":
            # Do not spend more requests after a known quota failure.
            break
    passed = sum(case["passed"] for case in cases)
    return {
        "mode": "live_generation",
        "model": config.model,
        "dataset_version": data["version"],
        "total": len(sources),
        "attempted": len(cases),
        "passed": passed,
        "metrics": {
            "valid_plan_rate": passed / len(sources),
            "median_latency_ms": statistics.median(case["latency_ms"] for case in cases),
        },
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Call Gemini using synthetic evidence only")
    parser.add_argument("--case", help="Run just one source ID in live mode, e.g. mixed_signs")
    parser.add_argument("--env-file", type=Path, help="Load a local ignored .env file for a live run")
    parser.add_argument("--output", type=Path, help="Also write the JSON report to this file")
    args = parser.parse_args()
    if (args.case or args.env_file) and not args.live:
        parser.error("--case and --env-file require --live")
    data = load_dataset()
    if args.case and args.case not in data["sources"]:
        parser.error(f"Unknown case. Choose one of: {', '.join(data['sources'])}")
    if args.env_file:
        if not args.env_file.is_file():
            parser.error(f"Environment file not found: {args.env_file}")
        load_dotenv(args.env_file)
    if args.live:
        try:
            config = GeminiConfig.from_env()
        except ExplanationError as exc:
            parser.error(str(exc))
        report = evaluate_live(data, config, args.case)
    else:
        report = evaluate_offline(data)
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
