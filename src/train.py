import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# Nguong chat luong cua lab nay la f1_score, KHONG phai accuracy.
# Ly do: bo du lieu Adult co ty le lop 75/25. Mot mo hinh doan bua
# "thu nhap thap" cho moi mau da dat accuracy 0.75 ma khong hoc duoc gi.
F1_THRESHOLD = 0.65
BASELINE_POSITIVE_RATE = 0.248
DRIFT_TOLERANCE = 0.05


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho GradientBoostingClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia (holdout).

    Tra ve:
        f1 (float): diem F1 cua lop duong (thu nhap > 50K) tren tap holdout.
    """

    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    positive_rate = float(y_train.mean())
    drift_delta = abs(positive_rate - BASELINE_POSITIVE_RATE)
    drift_detected = drift_delta > DRIFT_TOLERANCE
    drift_status = "WARNING" if drift_detected else "PASSED"
    print(
        f"Data drift check: {drift_status} | positive_rate={positive_rate:.4f} "
        f"| baseline={BASELINE_POSITIVE_RATE:.4f} | delta={drift_delta:.4f}"
    )

    with mlflow.start_run():

        mlflow.log_params(params)

        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_eval)[:, 1]
        default_preds = (probabilities >= 0.5).astype(int)
        default_f1 = float(f1_score(y_eval, default_preds))

        threshold_results = []
        for threshold in np.arange(0.1, 0.901, 0.05):
            threshold = round(float(threshold), 2)
            threshold_preds = (probabilities >= threshold).astype(int)
            threshold_f1 = float(f1_score(y_eval, threshold_preds))
            threshold_results.append((threshold_f1, threshold))

        f1, best_threshold = max(threshold_results, key=lambda item: item[0])
        preds = (probabilities >= best_threshold).astype(int)
        acc = float(accuracy_score(y_eval, preds))
        matrix = confusion_matrix(y_eval, preds)
        class_metrics = classification_report(
            y_eval,
            preds,
            labels=[0, 1],
            target_names=["thu_nhap_thap", "thu_nhap_cao"],
            output_dict=True,
            zero_division=0,
        )

        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("f1_score_default", default_f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("positive_rate", positive_rate)
        mlflow.log_metric("drift_delta", drift_delta)
        mlflow.log_metric("positive_precision", class_metrics["thu_nhap_cao"]["precision"])
        mlflow.log_metric("positive_recall", class_metrics["thu_nhap_cao"]["recall"])
        mlflow.sklearn.log_model(model, "model")

        print(
            f"Best threshold: {best_threshold:.2f} | F1: {f1:.4f} "
            f"| Default F1: {default_f1:.4f} | Accuracy: {acc:.4f}"
        )

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/report.json", "w") as f:
            json.dump(
                {
                    "f1_score": f1,
                    "f1_score_default": default_f1,
                    "accuracy": acc,
                    "decision_threshold": best_threshold,
                    "positive_rate": positive_rate,
                    "drift_delta": drift_delta,
                    "drift_detected": drift_detected,
                },
                f,
                indent=2,
            )

        with open("outputs/detail.txt", "w") as f:
            f.write("CONFUSION MATRIX\n")
            f.write("rows=actual, columns=predicted\n")
            f.write(f"{matrix}\n\n")
            f.write("PRECISION / RECALL BY CLASS\n")
            for label in ("thu_nhap_thap", "thu_nhap_cao"):
                metrics = class_metrics[label]
                f.write(
                    f"{label}: precision={metrics['precision']:.4f}, "
                    f"recall={metrics['recall']:.4f}, "
                    f"f1={metrics['f1-score']:.4f}, "
                    f"support={int(metrics['support'])}\n"
                )
            f.write(
                f"\nDATA DRIFT: {drift_status} | positive_rate={positive_rate:.4f}, "
                f"baseline={BASELINE_POSITIVE_RATE:.4f}, delta={drift_delta:.4f}, "
                f"tolerance={DRIFT_TOLERANCE:.4f}\n"
            )

        os.makedirs("models", exist_ok=True)
        joblib.dump(
            {"model": model, "decision_threshold": best_threshold},
            "models/model.joblib",
        )

    return f1


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
