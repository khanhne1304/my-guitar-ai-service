import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train clip-level guitar performance quality models."
    )
    parser.add_argument(
        "--config",
        default="config/training_config.yaml",
        help="Path tới file YAML cấu hình huấn luyện.",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def synthesize_dataset(cfg: Dict, dataset_path: Path, seed: int) -> pd.DataFrame:
    print(f"[INFO] Dataset chưa tồn tại, tạo synthetic data tại {dataset_path}")
    n_samples = int(cfg["num_samples"])
    rng = np.random.default_rng(seed)

    mean_pitch_error = rng.normal(loc=0.35, scale=0.25, size=n_samples)
    mean_pitch_error = np.clip(mean_pitch_error, 0.0, 2.5)
    std_pitch_error = np.clip(mean_pitch_error * rng.uniform(0.4, 1.2, n_samples), 0.05, 2.0)

    mean_timing_offset = rng.normal(loc=12, scale=18, size=n_samples)
    std_timing_offset = np.clip(rng.normal(35, 15, n_samples), 5, 120)

    onset_density = np.clip(rng.normal(2.5, 0.8, n_samples), 0.8, 4.5)
    tempo_variation_pct = np.clip(np.abs(rng.normal(3.0, 2.0, n_samples)), 0.2, 12.0)

    buzz_ratio = np.clip(rng.beta(1.5, 8, n_samples), 0.0, 0.6)
    missing_strings_ratio = np.clip(rng.beta(1.2, 6, n_samples), 0.0, 0.7)
    extra_noise_level = np.clip(rng.normal(0.2, 0.15, n_samples), 0.0, 1.0)
    mean_snr_db = np.clip(rng.normal(35, 8, n_samples), 10, 60)
    attack_smoothness = np.clip(rng.normal(0.65, 0.2, n_samples), 0.1, 1.0)
    sustain_consistency = np.clip(rng.normal(0.7, 0.18, n_samples), 0.05, 1.0)

    pitch_accuracy = np.clip(
        100
        - (mean_pitch_error * 18 + std_pitch_error * 8)
        + rng.normal(0, 3, n_samples),
        0,
        100,
    )
    timing_accuracy = np.clip(
        100 - (np.abs(mean_timing_offset) * 0.9 + std_timing_offset * 0.3)
        + rng.normal(0, 4, n_samples),
        0,
        100,
    )
    timing_stability = np.clip(
        100 - (std_timing_offset * 0.8) + rng.normal(0, 2, n_samples),
        0,
        100,
    )
    tempo_deviation_percent = tempo_variation_pct + rng.normal(0, 0.4, n_samples)
    tempo_deviation_percent = np.clip(tempo_deviation_percent, 0.1, 15.0)
    chord_cleanliness = np.clip(
        100
        - (buzz_ratio * 40 + missing_strings_ratio * 30 + extra_noise_level * 30)
        + rng.normal(0, 2.5, n_samples),
        0,
        100,
    )

    overall_score = (
        pitch_accuracy * 0.35
        + timing_accuracy * 0.3
        + chord_cleanliness * 0.2
        + (100 - tempo_deviation_percent * 3) * 0.15
    )
    overall_score = np.clip(overall_score / (0.35 + 0.3 + 0.2 + 0.15), 0, 100)

    level_class = np.digitize(overall_score, bins=[60, 80])

    df = pd.DataFrame(
        {
            "mean_pitch_error_semitones": mean_pitch_error,
            "std_pitch_error_semitones": std_pitch_error,
            "mean_timing_offset_ms": mean_timing_offset,
            "std_timing_offset_ms": std_timing_offset,
            "onset_density": onset_density,
            "tempo_variation_pct": tempo_variation_pct,
            "buzz_ratio": buzz_ratio,
            "missing_strings_ratio": missing_strings_ratio,
            "extra_noise_level": extra_noise_level,
            "mean_snr_db": mean_snr_db,
            "attack_smoothness": attack_smoothness,
            "sustain_consistency": sustain_consistency,
            "pitch_accuracy": pitch_accuracy,
            "timing_accuracy": timing_accuracy,
            "timing_stability": timing_stability,
            "tempo_deviation_percent": tempo_deviation_percent,
            "chord_cleanliness_score": chord_cleanliness,
            "overall_score": overall_score,
            "level_class": level_class,
        }
    )
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dataset_path, index=False)
    return df


def load_dataset(dataset_path: Path, cfg: Dict, seed: int) -> pd.DataFrame:
    if dataset_path.exists():
        print(f"[INFO] Tải dataset từ {dataset_path}")
        df = pd.read_csv(dataset_path)
        print(f"[INFO] Dataset có {len(df)} mẫu")
        return df

    # Kiểm tra xem có dataset Guitarset không
    guitarset_path = dataset_path.parent / "guitarset_metrics.csv"
    if guitarset_path.exists():
        print(f"[INFO] Tìm thấy dataset Guitarset tại {guitarset_path}")
        df = pd.read_csv(guitarset_path)
        print(f"[INFO] Dataset Guitarset có {len(df)} mẫu")
        return df

    synthetic_cfg = cfg.get("synthetic_dataset", {})
    if not synthetic_cfg.get("enabled", False):
        raise FileNotFoundError(
            f"Dataset không tồn tại tại {dataset_path} và synthetic_dataset.disabled. "
            f"Hãy chạy parse_guitarset_dataset.py để tạo dataset từ Guitarset."
        )
    print("[WARN] Sử dụng synthetic dataset. Khuyến nghị sử dụng dataset thật từ Guitarset.")
    return synthesize_dataset(synthetic_cfg, dataset_path, seed)


def split_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    reg_targets: List[str],
    cls_target: str,
    test_size: float,
    val_size: float,
    seed: int,
) -> Tuple:
    X = df[feature_cols]
    y_reg = df[reg_targets]
    y_cls = df[cls_target]

    stratify_vec = y_cls if len(np.unique(y_cls)) > 1 else None
    (
        X_trainval,
        X_test,
        y_reg_trainval,
        y_reg_test,
        y_cls_trainval,
        y_cls_test,
    ) = train_test_split(
        X,
        y_reg,
        y_cls,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_vec,
    )

    val_ratio = val_size / max(1e-8, (1 - test_size))
    val_ratio = np.clip(val_ratio, 0.05, 0.9)

    stratify_trainval = y_cls_trainval if len(np.unique(y_cls_trainval)) > 1 else None
    X_train, X_val, y_reg_train, y_reg_val, y_cls_train, y_cls_val = train_test_split(
        X_trainval,
        y_reg_trainval,
        y_cls_trainval,
        test_size=val_ratio,
        random_state=seed,
        stratify=stratify_trainval,
    )

    return (
        X_train,
        X_val,
        X_test,
        y_reg_train,
        y_reg_val,
        y_reg_test,
        y_cls_train,
        y_cls_val,
        y_cls_test,
    )


def train_regressor(X_train, y_train, seed: int) -> Pipeline:
    regressor = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=400,
                    max_depth=None,
                    min_samples_leaf=2,
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    regressor.fit(X_train, y_train)
    return regressor


def train_classifier(X_train, y_train, seed: int) -> Pipeline:
    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    clf.fit(X_train, y_train)
    return clf


def regression_metrics(y_true, y_pred, target_names: List[str]) -> Dict:
    metrics = {}
    for idx, name in enumerate(target_names):
        mse = mean_squared_error(y_true.iloc[:, idx], y_pred[:, idx])
        metrics[name] = {
            "mae": float(mean_absolute_error(y_true.iloc[:, idx], y_pred[:, idx])),
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(y_true.iloc[:, idx], y_pred[:, idx])),
        }
    metrics["macro_mae"] = float(mean_absolute_error(y_true, y_pred))
    metrics["macro_rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    metrics["macro_r2"] = float(r2_score(y_true, y_pred, multioutput="uniform_average"))
    return metrics


def classification_metrics(y_true, y_pred) -> Dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def save_metrics(path: Path, metrics: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_config(config_path)

    dataset_path = config_path.parent.parent / cfg["dataset_path"]
    artifacts_dir = config_path.parent.parent / cfg["artifacts_dir"]
    ensure_dirs(artifacts_dir)

    seed = int(cfg.get("random_seed", 42))

    df = load_dataset(dataset_path, cfg, seed)

    (
        X_train,
        X_val,
        X_test,
        y_reg_train,
        y_reg_val,
        y_reg_test,
        y_cls_train,
        y_cls_val,
        y_cls_test,
    ) = split_data(
        df,
        cfg["feature_columns"],
        cfg["regression_targets"],
        cfg["classification_target"],
        cfg["test_size"],
        cfg["val_size"],
        seed,
    )

    reg_model = train_regressor(X_train, y_reg_train, seed)
    cls_model = train_classifier(X_train, y_cls_train, seed)

    y_reg_val_pred = reg_model.predict(X_val)
    y_reg_test_pred = reg_model.predict(X_test)
    y_cls_val_pred = cls_model.predict(X_val)
    y_cls_test_pred = cls_model.predict(X_test)

    metrics = {
        "regression": {
            "val": regression_metrics(y_reg_val, y_reg_val_pred, cfg["regression_targets"]),
            "test": regression_metrics(
                y_reg_test, y_reg_test_pred, cfg["regression_targets"]
            ),
        },
        "classification": {
            "val": classification_metrics(y_cls_val, y_cls_val_pred),
            "test": classification_metrics(y_cls_test, y_cls_test_pred),
        },
        "config_snapshot": cfg,
    }

    joblib.dump(reg_model, artifacts_dir / "clip_regressor.joblib")
    joblib.dump(cls_model, artifacts_dir / "level_classifier.joblib")
    save_metrics(artifacts_dir / "metrics.json", metrics)

    print("[INFO] Huấn luyện hoàn tất. Metrics lưu tại artifacts/metrics.json")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

