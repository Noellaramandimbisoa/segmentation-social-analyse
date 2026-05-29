from sklearn.cluster import KMeans

def build_model(n_clusters=4, random_state=42):
    return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)