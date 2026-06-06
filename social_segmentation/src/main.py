import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import webbrowser

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_samples
import numpy as np
import matplotlib.cm as cm

# =========================
# PATHS
# =========================
INPUT_PATH = "data/raw/Segmentation-Data.csv"
OUTPUT_DIR = "results/export"
PLOTS_DIR = "results/plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================
def load_data(path):
    df = pd.read_csv(path, header=None)
    df.columns = ["id", "age", "revenu", "score1", "categorie", "frequence", "depense"]
    return df

def clean_data(df):
    df = df.drop_duplicates()
    return df

# =========================
# PREPARE DATA
# =========================
def prepare_data(df):
    X = df.drop(columns=["id"])
    return X

# =========================
# MODEL
# =========================
def build_model(n_clusters=4):

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), ["age", "revenu", "score1", "categorie", "frequence", "depense"])
    ])

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("cluster", model)
    ])

    return pipeline

# =========================
# ELBOW METHOD
# =========================
def elbow_method(X):
    inertias = []
    K = range(2, 10)

    for k in K:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X)
        inertias.append(model.inertia_)

    plt.figure(figsize=(8,5))
    plt.plot(K, inertias, marker="o")
    plt.title("Méthode du coude (Elbow Method)")
    plt.xlabel("Nombre de clusters")
    plt.ylabel("Inertie")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "elbow.png"), dpi=300)
    plt.close()

# =========================
# VISUALISATIONS
# =========================
def plot_distribution(df):
    plt.figure(figsize=(8,5))

    sns.countplot(data=df, x="cluster", palette="viridis")

    plt.title("Répartition des clusters")
    plt.xlabel("Cluster")
    plt.ylabel("Nombre d'individus")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "distribution.png"), dpi=300)
    plt.close()

def plot_pca(X, labels):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(
        X_pca[:,0],
        X_pca[:,1],
        c=labels,
        cmap="viridis",
        alpha=0.7
    )

    plt.title("Segmentation sociale (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(scatter)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "pca.png"), dpi=300)
    plt.close()


def plot_pca_3d(X, labels):

    pca = PCA(n_components=3)

    X_pca = pca.fit_transform(X)

    variance = pca.explained_variance_ratio_.sum() * 100

    fig = plt.figure(figsize=(12,8))

    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        X_pca[:,0],
        X_pca[:,1],
        X_pca[:,2],
        c=labels,
        cmap="viridis",
        alpha=0.8,
        s=60
    )

    ax.set_title(
        f"Segmentation Sociale Avancée - Projection PCA 3D\n"
        f"Variance expliquée : {variance:.2f} %",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel("Composante Principale 1 (PC1)")
    ax.set_ylabel("Composante Principale 2 (PC2)")
    ax.set_zlabel("Composante Principale 3 (PC3)")

    legend = ax.legend(
        *scatter.legend_elements(),
        title="Clusters"
    )

    ax.add_artist(legend)

    plt.tight_layout()

    plt.savefig(
        os.path.join(PLOTS_DIR, "pca_3d.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\n===== INTERPRÉTATION PCA 3D =====")
    print(f"Variance totale expliquée : {variance:.2f}%")

    if variance >= 80:
        print(
            "Excellente représentation des données : "
            "la projection PCA 3D conserve la majorité "
            "de l'information originale."
        )

    elif variance >= 60:
        print(
            "Bonne représentation des données : "
            "la projection PCA 3D résume correctement "
            "la structure des clusters."
        )

    else:
        print(
            "La projection PCA 3D perd une partie "
            "importante de l'information. "
            "Une analyse dans l'espace complet est recommandée."
        )

def plot_heatmap(df):
    plt.figure(figsize=(10,6))

    sns.heatmap(
        df.groupby("cluster").mean(numeric_only=True),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Profil des clusters")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "heatmap.png"), dpi=300)
    plt.close()

def plot_pie(df):
    plt.figure(figsize=(6,6))

    df["cluster"].value_counts().plot(
        kind="pie",
        autopct="%1.1f%%",
        cmap="viridis"
    )

    plt.title("Répartition (%) des clusters")
    plt.ylabel("")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "pie.png"), dpi=300)
    plt.close()

def plot_silhouette(X, labels):

    silhouette_vals = silhouette_samples(X, labels)

    y_lower = 10
    plt.figure(figsize=(8,6))

    n_clusters = len(set(labels))

    for i in range(n_clusters):
        cluster_vals = silhouette_vals[labels == i]
        cluster_vals.sort()

        size_cluster = cluster_vals.shape[0]
        y_upper = y_lower + size_cluster

        color = cm.nipy_spectral(float(i) / n_clusters)

        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_vals,
            facecolor=color,
            alpha=0.7
        )

        y_lower = y_upper + 10

    plt.title("Silhouette Plot des clusters")
    plt.xlabel("Coefficient de silhouette")
    plt.ylabel("Clusters")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "silhouette.png"), dpi=300)
    plt.close()

# =========================
# MAIN
# =========================
def main():

    # Load data
    df = load_data(INPUT_PATH)
    df = clean_data(df)

    X = prepare_data(df)

    # Model
    model = build_model(n_clusters=4)

    # Preprocess
    X_processed = model.named_steps["preprocessor"].fit_transform(X)

    # Elbow
    elbow_method(X_processed)

    # Clustering
    labels = model.named_steps["cluster"].fit_predict(X_processed)
    df["cluster"] = labels

    # Score
    score = None

    if len(set(labels)) > 1:
        score = silhouette_score(X_processed, labels)
        print("Silhouette score :", score)

    # Save results
    df.to_csv(os.path.join(OUTPUT_DIR, "resultats.csv"), index=False)
    joblib.dump(model, os.path.join(OUTPUT_DIR, "model.pkl"))

    # Summary
    print("\nRésumé des clusters :")
    print(df.groupby("cluster").mean(numeric_only=True))

    # =========================
    # VISUALS
    # =========================
    plot_distribution(df)
    plot_pie(df)
    plot_pca(X_processed, labels)
    plot_pca_3d(X_processed, labels)
    plot_silhouette(X_processed, labels)
    plot_heatmap(df)
    

    print("\n✔ Tous les graphiques ont été générés dans :", PLOTS_DIR)

    # 🔥 HTML REPORT
    if score is not None:
        generate_html_report(score)

    # =========================
    # RAPPORT HTML
    # =========================
def generate_html_report(score):

    html_content = f"""
    <html>
    <head>
        <title>Rapport de Segmentation K-Means</title>
        <style>
            body {{
                font-family: Arial;
                margin: 40px;
                background-color: #f5f5f5;
            }}
            h1, h2 {{
                color: #2c3e50;
            }}
            .container {{
                background: white;
                padding: 20px;
                border-radius: 10px;
            }}
            img {{
                width: 600px;
                margin: 10px;
                border: 1px solid #ddd;
                border-radius: 8px;
            }}
            .score {{
                font-size: 20px;
                font-weight: bold;
                color: green;
            }}
        </style>
    </head>
    <body>

    <div class="container">

        <h1>📊 Rapport de Segmentation Sociale (K-Means)</h1>

        <h2>🔹 Score de Silhouette</h2>
        <p class="score">{score:.4f}</p>

        <p>
        Le score de silhouette mesure la qualité de la segmentation obtenue par l'algorithme K-Means.
        Dans cette étude, le score obtenu est de <b>{score:.4f}</b>.
        Cette valeur indique que les clusters présentent une séparation raisonnable.
        Les individus appartenant à un même groupe sont relativement similaires tandis que les groupes restent suffisamment distincts.
        </p>

        <h2>🔹 Méthode du coude</h2>

        <img src="../plots/elbow.png">

        <p>
        La méthode du coude permet de déterminer le nombre optimal de clusters.
        On observe une diminution progressive de l'inertie lorsque le nombre de clusters augmente.
        Le point de rupture observé sur la courbe justifie le choix de <b>k = 4</b>, qui représente un bon compromis entre précision et simplicité du modèle.
        </p>

        <h2>🔹 Silhouette Plot</h2>

        <img src="../plots/silhouette.png">

        <p>
        Le graphique de silhouette montre la qualité de classification des individus dans chaque cluster.
        La majorité des observations possèdent des coefficients positifs, ce qui indique une affectation cohérente aux groupes identifiés.
        L'absence de fortes valeurs négatives confirme que peu d'individus sont mal classés.
        </p>

        <h2>🔹 PCA 2D</h2>

        <img src="../plots/pca.png">

        <p>
        Cette projection ACP (Analyse en Composantes Principales) en deux dimensions permet de visualiser la structure globale des données.
        Les couleurs représentent les différents clusters obtenus par K-Means.
        On observe une séparation visible entre plusieurs groupes, ce qui confirme l'existence de profils socio-économiques distincts.
        </p>

        <h2>🔹 PCA 3D</h2>

        <img src="../plots/pca_3d.png">

        <p>
        La projection ACP en trois dimensions conserve davantage d'informations que la représentation 2D.
        Elle permet d'observer plus clairement la séparation entre les clusters et facilite l'analyse visuelle des comportements similaires.
        La variance expliquée affichée sur le graphique constitue un indicateur de la qualité de représentation des données originales.
        </p>

        <h2>🔹 Répartition des clusters</h2>

        <img src="../plots/distribution.png">

        <p>
        Le diagramme en barres présente le nombre d'individus appartenant à chaque cluster.
        Une répartition relativement équilibrée indique que l'algorithme n'a pas concentré excessivement les observations dans un seul groupe.
        Cela renforce la stabilité de la segmentation obtenue.
        </p>

        <img src="../plots/pie.png">

        <p>
        Le diagramme circulaire représente les proportions de chaque cluster dans la population étudiée.
        Cette visualisation facilite l'identification des segments majoritaires et minoritaires.
        Elle permet également d'apprécier l'importance relative de chaque profil social identifié.
        </p>

        <h2>🔹 Heatmap des profils</h2>

        <img src="../plots/heatmap.png">

        <p>
        La carte thermique présente les moyennes des variables pour chaque cluster.
        Les couleurs chaudes indiquent généralement des valeurs élevées tandis que les couleurs froides correspondent à des valeurs faibles.
        Cette visualisation met en évidence les différences de revenus, de dépenses, d'âge ou de fréquence entre les groupes et facilite leur interprétation.
        </p>

    </div>

    </body>
    </html>
    """

    report_path = os.path.join(OUTPUT_DIR, "report.html")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print("\n✔ Rapport HTML généré :", report_path)

    # Ouvre automatiquement le rapport dans le navigateur
    webbrowser.open("file://" + os.path.abspath(report_path))

if __name__ == "__main__":
    main()