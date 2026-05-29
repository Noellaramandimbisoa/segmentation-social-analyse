import os
import joblib
import pandas as pd

from config import Config
from data_loader import load_data
from preprocess import clean_data
from features import build_scaler
from model import build_model
from evaluation import evaluate_clustering
from visualization import plot_cluster_distribution, plot_feature_distributions

def main():
    cfg = Config()
    cfg.ensure_dirs()

    df = load_data(cfg.paths["raw_data"])
    df = clean_data(df)

    processed_path = cfg.paths["processed_data"]
    df.to_csv(processed_path, index=False)

    X = df.drop(columns=["id"], errors="ignore")

    scaler = build_scaler()
    X_scaled = scaler.fit_transform(X)

    model = build_model(
        n_clusters=cfg.data["n_clusters"],
        random_state=cfg.data["random_state"]
    )

    labels = model.fit_predict(X_scaled)

    metrics = evaluate_clustering(X_scaled, labels)
    print(metrics)

    df["cluster"] = labels
    df.to_csv(os.path.join(cfg.paths["output_dir"], "segmented_data.csv"), index=False)

    joblib.dump(model, cfg.paths["model_path"])
    joblib.dump(scaler, cfg.paths["scaler_path"])

    plot_cluster_distribution(df, os.path.join(cfg.paths["plots_dir"], "cluster_distribution.png"))
    plot_feature_distributions(df, cfg.paths["plots_dir"])


    joblib.dump(model, cfg.paths["model_path"])
    joblib.dump(scaler, cfg.paths["scaler_path"])

    loaded_model = joblib.load(cfg.paths["model_path"])
    print(loaded_model)

if __name__ == "__main__":
    main()