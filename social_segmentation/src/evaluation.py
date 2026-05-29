from sklearn.metrics import silhouette_score, davies_bouldin_score

def evaluate_clustering(X, labels):
    result = {}
    if len(set(labels)) > 1:
        result["silhouette_score"] = silhouette_score(X, labels)
        result["davies_bouldin_score"] = davies_bouldin_score(X, labels)
    else:
        result["silhouette_score"] = None
        result["davies_bouldin_score"] = None
    return result