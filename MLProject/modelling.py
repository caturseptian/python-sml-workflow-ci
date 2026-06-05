"""
MLProject/modelling.py  (Kriteria 3 - dijalankan via MLflow Project)
====================================================================
Versi modelling untuk MLflow Project. Dipanggil oleh `mlflow run .` melalui
GitHub Actions. Menerima parameter dari MLProject, melatih Random Forest pada
dataset hasil preprocessing, dan me-log model + metrik ke MLflow.

Saat dijalankan di CI, tracking diarahkan ke local file store (folder ./mlruns)
sehingga artefak bisa diunggah sebagai GitHub artifact, dan run_id dipakai untuk
`mlflow models build-docker` (advanced).
"""

import argparse
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

TARGET_COLUMN = "diagnosis"


def load_dataset(data_path: str):
    train = pd.read_csv(os.path.join(data_path, "train.csv"))
    test = pd.read_csv(os.path.join(data_path, "test.csv"))
    X_train = train.drop(columns=[TARGET_COLUMN])
    y_train = train[TARGET_COLUMN]
    X_test = test.drop(columns=[TARGET_COLUMN])
    y_test = test[TARGET_COLUMN]
    return X_train, y_train, X_test, y_test


def main(args):
    X_train, y_train, X_test, y_test = load_dataset(args.data_path)

    # autolog mencatat params & model secara otomatis
    mlflow.sklearn.autolog(log_models=True)

    # Saat dijalankan via `mlflow run`, MLflow sudah menyiapkan run melalui
    # environment variable MLFLOW_RUN_ID -> cukup attach (start_run tanpa arg).
    # Saat dijalankan langsung (`python modelling.py`), buat experiment & run baru.
    env_run_id = os.environ.get("MLFLOW_RUN_ID")
    if env_run_id:
        active = mlflow.start_run()
    else:
        mlflow.set_experiment("CI - Breast Cancer Classification")
        active = mlflow.start_run(run_name="ci_random_forest")
    run_id = active.info.run_id

    try:
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=(None if args.max_depth < 0 else args.max_depth),
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        # metrik tambahan (manual) di samping autolog
        mlflow.log_metric("test_accuracy", accuracy_score(y_test, preds))
        mlflow.log_metric("test_precision", precision_score(y_test, preds))
        mlflow.log_metric("test_recall", recall_score(y_test, preds))
        mlflow.log_metric("test_f1", f1_score(y_test, preds))
        mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, proba))

        # pastikan model ter-log di artifact_path "model" untuk build-docker
        mlflow.sklearn.log_model(model, artifact_path="model")

        # tulis run_id agar mudah dibaca step CI berikutnya
        with open("run_id.txt", "w") as f:
            f.write(run_id)
        print("RUN_ID:", run_id)
        print("Akurasi test:", accuracy_score(y_test, preds))
    finally:
        mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="breast_cancer_preprocessing")
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=-1, help="-1 berarti None")
    main(parser.parse_args())
