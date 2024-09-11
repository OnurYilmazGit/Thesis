import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

# Benchmark class
class Benchmark:
    def __init__(self, X_full, X_core, y_full, y_core, class_names):
        self.X_full = X_full
        self.X_core = X_core
        self.y_full = y_full
        self.y_core = y_core
        self.class_names = class_names

    def jensen_shannon_divergence(self):
        """
        Compute the Jensen-Shannon Divergence for each feature in the real and core datasets.
        """
        num_features = self.X_full.shape[1]  # Number of features
        js_divergences = []

        for col in range(num_features):
            p = self.X_full[:, col]  # Full dataset feature column
            q = self.X_core[:, col]  # Core dataset feature column
            
            # Calculate the histograms to estimate distributions (assuming continuous data)
            p_hist, _ = np.histogram(p, bins=50, density=True)
            q_hist, _ = np.histogram(q, bins=50, density=True)

            # Ensure no zero entries in the histograms
            p_hist += 1e-8
            q_hist += 1e-8

            # Calculate the Jensen-Shannon divergence between the two histograms
            js_div = jensenshannon(p_hist, q_hist)
            js_divergences.append(js_div)

            print(f"Feature {col}: Jensen-Shannon Divergence = {js_div:.4f}")

        avg_js_div = np.mean(js_divergences)
        print(f"\nAverage Jensen-Shannon Divergence across all features: {avg_js_div:.4f}")

    def compare_statistics(self):
        """
        Compare the statistical properties (mean and variance) between the real and synthetic datasets.
        """
        # Mean and variance of the full dataset
        full_mean = np.mean(self.X_full, axis=0)
        full_variance = np.var(self.X_full, axis=0)

        # Mean and variance of the core dataset
        core_mean = np.mean(self.X_core, axis=0)
        core_variance = np.var(self.X_core, axis=0)

        print("\n=== Statistical Comparison ===")
        print(f"Mean (Full Data): {full_mean}")
        print(f"Variance (Full Data): {full_variance}")
        print(f"Mean (Core Set): {core_mean}")
        print(f"Variance (Core Set): {core_variance}")

        # Compare overall mean and variance differences
        mean_diff = np.mean(np.abs(full_mean - core_mean))
        variance_diff = np.mean(np.abs(full_variance - core_variance))

        print(f"\nOverall Mean Difference: {mean_diff:.4f}")
        print(f"Overall Variance Difference: {variance_diff:.4f}")


