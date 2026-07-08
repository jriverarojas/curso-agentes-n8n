import argparse
import json
import math
import tempfile
from pathlib import Path

import boto3
import numpy as np


RISK_FEATURES = [
    "debt_to_income_ratio",
    "loan_to_value_ratio",
    "payment_to_income_ratio",
    "expense_to_income_ratio",
    "total_obligations_to_income_ratio",
    "employment_stability_score",
    "banking_capacity_score",
    "credit_history_score",
]

AMOUNT_FEATURES = [
    "net_monthly_income",
    "monthly_debt_payment",
    "monthly_expenses",
    "property_value",
    "requested_amount",
    "requested_term_months",
    "estimated_monthly_payment",
    "debt_to_income_ratio",
    "loan_to_value_ratio",
    "payment_to_income_ratio",
    "expense_to_income_ratio",
    "total_obligations_to_income_ratio",
    "employment_stability_score",
    "banking_capacity_score",
    "credit_history_score",
]


def read_json_file(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_json_s3(bucket, key):
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(response["Body"].read().decode("utf-8"))


def download_s3_file(bucket, key, destination):
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, destination)


def sigmoid(value):
    return 1 / (1 + math.exp(-value))


def predict_risk_probabilities_from_json(model, rows):
    probabilities = []

    for _, row in rows.iterrows():
        score = float(model["intercept"])
        for feature_name in model["features"]:
            raw_value = float(row[feature_name])
            mean = float(model["scaler"]["mean"][feature_name])
            scale = float(model["scaler"]["scale"].get(feature_name) or 1)
            coefficient = float(model["coefficients"][feature_name])
            standardized_value = (raw_value - mean) / scale
            score += standardized_value * coefficient

        probabilities.append(sigmoid(score))

    probabilities = np.array(probabilities)
    return np.column_stack([1 - probabilities, probabilities])


def explain_risk(features, model):
    score = model["intercept"]
    explanation = []

    for feature_name in model["features"]:
        raw_value = float(features[feature_name])
        mean = float(model["scaler"]["mean"][feature_name])
        scale = float(model["scaler"]["scale"].get(feature_name) or 1)
        coefficient = float(model["coefficients"][feature_name])
        standardized_value = (raw_value - mean) / scale
        contribution = standardized_value * coefficient
        score += contribution

        explanation.append(
            {
                "feature": feature_name,
                "value": raw_value,
                "standardized_value": round(standardized_value, 6),
                "coefficient": round(coefficient, 6),
                "contribution": round(contribution, 6),
                "direction": "sube el riesgo" if contribution > 0 else "baja el riesgo",
            }
        )

    probability = sigmoid(score)
    threshold = float(model.get("threshold", 0.5))
    explanation.sort(key=lambda item: abs(item["contribution"]), reverse=True)

    return {
        "default_probability": round(probability, 4),
        "risk_label": "HIGH" if probability >= threshold else "LOW",
        "threshold": threshold,
        "score": round(score, 6),
        "base_score": round(float(model["intercept"]), 6),
        "contributions_sum": round(float(np.sum([item["contribution"] for item in explanation])), 6),
        "risk_explanation": explanation,
    }


def explain_risk_with_shap(features, model, background_rows):
    import pandas as pd
    import shap

    background = pd.DataFrame(background_rows)[model["features"]]
    case = pd.DataFrame([{name: features[name] for name in model["features"]}])

    def predict_proba(rows):
        frame = pd.DataFrame(rows, columns=model["features"])
        return predict_risk_probabilities_from_json(model, frame)

    explainer = shap.Explainer(predict_proba, background)
    shap_result = explainer(case)

    values = shap_result.values
    base_values = shap_result.base_values

    if values.ndim == 3:
        contributions = values[0, :, 1]
        base_value = float(base_values[0][1])
    else:
        contributions = values[0]
        base_value = float(base_values[0])

    probability = float(predict_risk_probabilities_from_json(model, case)[0][1])
    threshold = float(model.get("threshold", 0.5))

    explanation = [
        {
            "feature": feature,
            "value": float(case.iloc[0][feature]),
            "contribution": round(float(contribution), 6),
            "direction": "sube el riesgo" if contribution > 0 else "baja el riesgo",
        }
        for feature, contribution in zip(model["features"], contributions)
    ]
    explanation.sort(key=lambda item: abs(item["contribution"]), reverse=True)

    contributions_sum = float(np.sum(contributions))

    return {
        "default_probability": round(probability, 4),
        "risk_label": "HIGH" if probability >= threshold else "LOW",
        "threshold": threshold,
        "base_value": round(base_value, 6),
        "contributions_sum": round(contributions_sum, 6),
        "reconstructed_probability": round(base_value + contributions_sum, 6),
        "method": "shap_with_background",
        "risk_explanation": explanation,
    }


def evaluate_tree_with_path(node, features):
    path = []
    current = node

    while "leaf" not in current:
        split = current["split"]
        condition = float(current.get("split_condition", 0))
        raw_value = float(features[split])
        next_node_id = current["yes"] if raw_value < condition else current["no"]

        path.append(
            {
                "feature": split,
                "value": raw_value,
                "condition": condition,
                "decision": "<" if raw_value < condition else ">=",
            }
        )

        children = current.get("children", [])
        current = next(child for child in children if child["nodeid"] == next_node_id)

    return float(current["leaf"]), path


def explain_amount(features, model):
    prediction = float(model["base_score"])
    contributions_by_feature = {}

    for tree in model["trees"]:
        leaf_value, path = evaluate_tree_with_path(tree, features)
        prediction += leaf_value

        if not path:
            continue

        split_share = leaf_value / len(path)
        for step in path:
            feature = step["feature"]
            contributions_by_feature.setdefault(
                feature,
                {
                    "feature": feature,
                    "value": float(features[feature]),
                    "contribution": 0.0,
                    "times_used_in_tree_paths": 0,
                },
            )
            contributions_by_feature[feature]["contribution"] += split_share
            contributions_by_feature[feature]["times_used_in_tree_paths"] += 1

    clip = model.get("prediction_clip") or {}
    if "min" in clip:
        prediction = max(float(clip["min"]), prediction)
    if "max" in clip:
        prediction = min(float(clip["max"]), prediction)

    explanation = []
    for item in contributions_by_feature.values():
        contribution = float(item["contribution"])
        explanation.append(
            {
                "feature": item["feature"],
                "value": item["value"],
                "contribution": round(contribution, 2),
                "times_used_in_tree_paths": item["times_used_in_tree_paths"],
                "direction": (
                    "sube el monto recomendado"
                    if contribution > 0
                    else "baja el monto recomendado"
                ),
            }
        )

    explanation.sort(key=lambda item: abs(item["contribution"]), reverse=True)

    return {
        "requested_amount": float(features.get("requested_amount", 0)),
        "recommended_amount": round(prediction, 2),
        "base_score": round(float(model["base_score"]), 2),
        "contributions_sum": round(
            float(sum(item["contribution"] for item in explanation)),
            2,
        ),
        "method": "tree_path_contribution_approximation",
        "amount_explanation": explanation,
    }


def explain_amount_with_xgboost(features, bucket, native_model_key):
    try:
        import pandas as pd
        import xgboost as xgb
    except Exception as error:
        raise SystemExit(
            "ERROR_XGBOOST_RUNTIME: no se pudo cargar XGBoost. "
            "En macOS instala OpenMP con `brew install libomp` y luego vuelve a ejecutar. "
            "En Windows verifica Docker/Visual C++ runtime o usa el entorno de SageMaker."
        )

    case = pd.DataFrame([{name: features[name] for name in AMOUNT_FEATURES}])

    with tempfile.NamedTemporaryFile(suffix=".json") as model_file:
        download_s3_file(bucket, native_model_key, model_file.name)
        booster = xgb.Booster()
        booster.load_model(model_file.name)

        matrix = xgb.DMatrix(case, feature_names=AMOUNT_FEATURES)
        contributions = booster.predict(matrix, pred_contribs=True)[0]

    base_value = float(contributions[-1])
    feature_contributions = contributions[:-1]
    prediction = float(base_value + np.sum(feature_contributions))

    explanation = [
        {
            "feature": feature,
            "value": float(case.iloc[0][feature]),
            "contribution": round(float(contribution), 2),
            "direction": (
                "sube el monto recomendado"
                if contribution > 0
                else "baja el monto recomendado"
            ),
        }
        for feature, contribution in zip(AMOUNT_FEATURES, feature_contributions)
    ]
    explanation.sort(key=lambda item: abs(item["contribution"]), reverse=True)

    return {
        "requested_amount": float(features.get("requested_amount", 0)),
        "recommended_amount": round(prediction, 2),
        "base_value": round(base_value, 2),
        "contributions_sum": round(float(np.sum(feature_contributions)), 2),
        "method": "xgboost_pred_contribs",
        "amount_explanation": explanation,
    }


def build_local_explanation(args):
    payload = read_json_file(args.input)
    application_id = payload["application_id"]
    features = payload["features"]
    risk_model = read_json_s3(args.bucket, args.risk_model_key)
    amount_model = read_json_s3(args.bucket, args.amount_model_key)

    risk_features = {name: features[name] for name in RISK_FEATURES}
    amount_features = {name: features[name] for name in AMOUNT_FEATURES}

    if args.risk_background_key:
        risk_background = read_json_s3(args.bucket, args.risk_background_key)
        risk = explain_risk_with_shap(risk_features, risk_model, risk_background)
    else:
        risk = explain_risk(risk_features, risk_model)

    amount = explain_amount(amount_features, amount_model)

    return {
        "application_id": application_id,
        "risk": {
            "default_probability": risk["default_probability"],
            "risk_label": risk["risk_label"],
            "threshold": risk["threshold"],
        },
        "risk_explanation_summary": {
            "base_value": risk.get("base_value", risk.get("base_score")),
            "contributions_sum": risk["contributions_sum"],
            "reconstructed_probability": risk.get("reconstructed_probability"),
            "score": risk.get("score"),
            "method": risk.get("method", "logistic_coefficient_contributions"),
        },
        "risk_explanation": risk["risk_explanation"][:8],
        "amount": {
            "requested_amount": amount["requested_amount"],
            "recommended_amount": amount["recommended_amount"],
        },
        "amount_explanation_summary": {
            "base_value": amount.get("base_value", amount.get("base_score")),
            "contributions_sum": amount.get("contributions_sum"),
            "method": amount.get("method", "tree_path_contribution_approximation"),
        },
        "amount_explanation": amount["amount_explanation"][:8],
    }


def read_global_explanation(args):
    return read_json_s3(args.bucket, args.key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["local", "read-global"], required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--key")
    parser.add_argument("--input")
    parser.add_argument("--risk-model-key")
    parser.add_argument("--amount-model-key")
    parser.add_argument("--risk-background-key")
    parser.add_argument("--amount-native-model-key")
    args = parser.parse_args()

    if args.mode == "read-global":
        result = read_global_explanation(args)
    else:
        result = build_local_explanation(args)

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
