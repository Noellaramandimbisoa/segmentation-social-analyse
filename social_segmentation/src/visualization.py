import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_cluster_distribution(df, output_path):
    plt.figure(figsize=(8, 5))
    sns.countplot(x="cluster", data=df, palette="viridis")
    plt.title("Répartition des clusters")
    plt.xlabel("Cluster")
    plt.ylabel("Nombre d'individus")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_feature_distributions(df, output_dir):
    numeric_cols = [c for c in df.columns if c not in ["id", "cluster"]]
    for col in numeric_cols:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True, color="steelblue")
        plt.title(f"Distribution de {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{col}_hist.png"), dpi=300)
        plt.close()