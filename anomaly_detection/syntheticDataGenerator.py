from sklearn.cluster import KMeans
import numpy as np

class SyntheticDataGenerator:
    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters
    
    def generate_synthetic_data(self, X_pca):
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        kmeans.fit(X_pca)
        
        # Cluster centers will serve as synthetic data points
        synthetic_data = kmeans.cluster_centers_
        return synthetic_data
    
    def evaluate_clustering(self, X_pca):
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        labels = kmeans.fit_predict(X_pca)
        
        # Evaluate the clustering performance using metrics
        db_score = davies_bouldin_score(X_pca, labels)
        ch_score = calinski_harabasz_score(X_pca, labels)
        
        return db_score, ch_score
