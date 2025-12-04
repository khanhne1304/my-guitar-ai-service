import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
import yaml

# Bỏ qua cảnh báo feature names giữa pandas/numpy trong pipeline sklearn
warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference for clip-level guitar practice scoring."
    )
    parser.add_argument(
        "--config",
        default="config/training_config.yaml",
        help="Path tới file cấu hình YAML dùng để xác định feature columns.",
    )
    parser.add_argument(
        "--regressor",
        default="artifacts/clip_regressor.joblib",
        help="Đường dẫn tới mô hình regression (joblib pipeline).",
    )
    parser.add_argument(
        "--classifier",
        default="artifacts/level_classifier.joblib",
        help="Đường dẫn tới mô hình classification (joblib pipeline).",
    )
    parser.add_argument(
        "--input",
        help="Đường dẫn file JSON chứa payload. Nếu bỏ trống sẽ đọc từ STDIN.",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_payload(input_path: Path | None) -> Dict[str, Any]:
    if input_path:
        raw = input_path.read_text(encoding="utf-8")
    else:
        raw = sys.stdin.read()
    if not raw.strip():
        raise ValueError("Không nhận được payload JSON nào.")
    return json.loads(raw)


def ensure_features(
    features: Dict[str, Any], feature_columns: list[str]
) -> pd.DataFrame:
    missing = [col for col in feature_columns if col not in features]
    if missing:
        raise ValueError(f"Thiếu các trường đặc trưng: {', '.join(missing)}")
    row = {}
    for col in feature_columns:
        value = features[col]
        if isinstance(value, (int, float)) and np.isfinite(value):
            row[col] = float(value)
            continue
        try:
            casted = float(value)
            if not np.isfinite(casted):
                raise ValueError
            row[col] = casted
        except Exception as exc:  # pylint: disable=broad-except
            raise ValueError(f"Giá trị của {col} không hợp lệ: {value}") from exc
    return pd.DataFrame([row], columns=feature_columns)


def predict_scores(
    reg_model_path: Path,
    cls_model_path: Path,
    features_df: pd.DataFrame,
    reg_targets: list[str],
) -> Dict[str, Any]:
    reg_model = joblib.load(reg_model_path)
    cls_model = joblib.load(cls_model_path)

    feature_matrix = features_df.to_numpy(dtype=float)

    reg_pred = reg_model.predict(feature_matrix)
    cls_pred = cls_model.predict(feature_matrix)

    reg_output = dict(zip(reg_targets, reg_pred[0].tolist()))

    cls_probs = None
    model_stage = getattr(cls_model, "named_steps", None)
    if model_stage and "model" in model_stage:
        estimator = model_stage["model"]
        if hasattr(estimator, "predict_proba"):
            cls_probs = estimator.predict_proba(feature_matrix)[0].tolist()

    cls_output = {
        "level_class": int(cls_pred[0]),
    }
    if cls_probs is not None:
        cls_output["probabilities"] = cls_probs

    return {
        "regression": reg_output,
        "classification": cls_output,
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    reg_path = Path(args.regressor)
    cls_path = Path(args.classifier)
    input_path = Path(args.input) if args.input else None

    try:
        cfg = load_config(config_path)
        payload = read_payload(input_path)
        features = payload.get("features")
        metadata = payload.get("metadata", {})
        if not isinstance(features, dict):
            raise ValueError("Payload phải chứa object 'features'.")

        feature_cols = cfg["feature_columns"]
        features_df = ensure_features(features, feature_cols)
        scores = predict_scores(
            reg_path,
            cls_path,
            features_df,
            cfg["regression_targets"],
        )

        output = {
            "success": True,
            "scores": scores,
            "metadata": metadata,
        }
        print(json.dumps(output, ensure_ascii=False))
    except Exception as exc:  # pylint: disable=broad-except
        error_payload = {
            "success": False,
            "error": str(exc),
        }
        print(json.dumps(error_payload, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

