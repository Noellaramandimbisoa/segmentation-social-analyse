import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer

INPUT_PATH = "data/raw/Segmentation-Data.csv"
OUTPUT_DIR = "results/export"
PLOTS_DIR = "results/plots"
MODEL_PATH = os.path.join(OUTPUT_DIR, "segmentation_pipeline.pkl")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "segmented_data.csv")
CENTERS_CSV = os.path.join(OUTPUT_DIR, "cluster_summary.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_data(path):
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def clean_data(df):
    df = df.copy()
    df = df.drop_duplicates()
    return df

def build_pipeline(n_clusters=4):
    numeric_features = []
    categorical_features = []

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric_features),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), categorical_features),
        ],
        remainder="drop"
    )

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("cluster", model)
    ])
    return pipeline

def prepare_columns(df):
    drop_cols = [col for col in ["id", "cluster"] if col in df.columns]
    feature_df = df.drop(columns=drop_cols, errors="ignore")

    numeric_features = feature_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = feature_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    return feature_df, numeric_features, categorical_features

def set_pipeline_columns(pipeline, numeric_features, categorical_features):
    pipeline.named_steps["preprocessor"].transformers = [
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_features),
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), categorical_features),
    ]
    return pipeline

def choose_k(X_processed, k_min=2, k_max=10):
    inertias = []
    ks = range(k_min, k_max + 1)
    for k in ks:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X_processed)
        inertias.append(model.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(list(ks), inertias, marker="o")
    plt.xlabel("Nombre de clusters")
    plt.ylabel("Inertie")
    plt.title("Méthode du coude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "elbow_method.png"), dpi=300)
    plt.close()

def plot_cluster_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(x="cluster", data=df, palette="viridis")
    plt.title("Répartition des clusters")
    plt.xlabel("Cluster")
    plt.ylabel("Nombre d'individus")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cluster_distribution.png"), dpi=300)
    plt.close()

def main():
    df = load_data(INPUT_PATH)
    print("Colonnes détectées :", df.columns.tolist())

    df = clean_data(df)

    feature_df, numeric_features, categorical_features = prepare_columns(df)
    print("Colonnes numériques :", numeric_features)
    print("Colonnes catégorielles :", categorical_features)

    pipeline = build_pipeline(n_clusters=4)
    pipeline = set_pipeline_columns(pipeline, numeric_features, categorical_features)

    X_processed = pipeline.named_steps["preprocessor"].fit_transform(feature_df)

    choose_k(X_processed, 2, 10)

    labels = pipeline.named_steps["cluster"].fit_predict(X_processed)
    df["cluster"] = labels

    if len(set(labels)) > 1:
        score = silhouette_score(X_processed, labels)
    else:
        score = None

    print("Silhouette score :", score)

    df.to_csv(OUTPUT_CSV, index=False)
    joblib.dump(pipeline, MODEL_PATH)

    summary = df.groupby("cluster").mean(numeric_only=True)
    summary.to_csv(CENTERS_CSV)

    plot_cluster_distribution(df)

    print("\nRésumé par cluster :")
    print(summary)

if __name__ == "__main__":
    main()