import os
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Import custom classes
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from visualization import Visualization
import numpy as np

# Suppress FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    # Define paths
    responses_path = '../responses'
    sensors_path = '../sensors'
    nodes = [f'node{i}' for i in range(6)]  # Adjusted node count

    # Step 1: Load and preprocess data
    print("Loading response and sensor data...")
    data_loader = DataLoader(responses_path, sensors_path, nodes)
    responses = data_loader.load_responses()
    sensor_data = data_loader.load_sensors()
    print("Data loaded successfully.")

    print("Merging response and sensor data...")
    data = pd.merge(sensor_data, responses, on=['Time', 'Node'])
    data['Value'] = data['Value'].map({label: idx for idx, label in enumerate(['Kripke', 'AMG', 'PENNANT', 'linpack', 'LAMMPS', 'Quicksilver'])})
    print("Data merged successfully.")

    # Step 2: Feature engineering and normalization
    print("Performing feature engineering and data normalization...")
    preprocessor = DataPreprocessor(data, window_size=20)
    data = preprocessor.feature_engineering()
    
    # Define the feature columns for normalization (excluding non-feature columns)
    feature_columns = [col for col in data.columns if col not in ['Time', 'Node', 'Value', 'Cluster']]
    X_scaled = preprocessor.normalize_data(feature_columns)

    print(f"Original number of features: {X_scaled.shape[1]}")
    print(f"Original data size in memory: {X_scaled.nbytes / 1024:.2f} KB")

    # Step 3: Apply PCA with optimal number of dimensions
    print("Applying PCA with optimal dimensions...")
    X_pca, pca_model = preprocessor.apply_pca(X_scaled, n_components=None)  # Auto-select components by variance
    
    # Determine the optimal number of components to retain a high variance
    optimal_components = next(i for i, cumulative_variance in enumerate(np.cumsum(pca_model.explained_variance_ratio_)) if cumulative_variance > 0.91)
    print(f"Optimal number of PCA components: {optimal_components}")
    
    # Re-run PCA with the optimal number of components
    X_pca, pca_model = preprocessor.apply_pca(X_scaled, n_components=optimal_components)
    print("PCA dimensionality reduction completed.")

    # Step 4: Train and evaluate Random Forest on the full PCA-reduced dataset
    print("Training Random Forest model on PCA-reduced dataset...")
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_pca, data['Value'], test_size=0.2, random_state=42)
    
    rf_model = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
    rf_model.fit(X_train_split, y_train_split)
    
    # Predictions on validation set
    y_pred = rf_model.predict(X_val_split)
    
    # Evaluation metrics
    print("Model performance on validation set (PCA-reduced dataset):")
    print(confusion_matrix(y_val_split, y_pred))
    print(classification_report(y_val_split, y_pred))
    print(f"Accuracy: {accuracy_score(y_val_split, y_pred)}")

    print(f"Reduced number of features after PCA: {X_pca.shape[1]}")
    print(f"Reduced data size in memory after PCA: {X_pca.nbytes / 1024:.2f} KB")

    # Step 5: Apply K-Means clustering for core set selection with more clusters and points
    print("Applying K-Means clustering for core set selection...")

    # Increase clusters and points per cluster to handle larger dataset
    optimal_k = preprocessor.optimize_cluster_selection(X_pca, max_clusters=300)  # Increase the number of clusters
    core_set = preprocessor.refine_cluster_selection(X_pca, n_clusters=optimal_k, points_per_cluster=10)  # Increase points per cluster
 
    X_core_pca = X_pca[core_set.index]
    y_core = core_set['Value']

    # Step 6: Re-train and evaluate Random Forest on the core set
    print("Training Random Forest model on the core set...")
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_core_pca, y_core, test_size=0.2, random_state=42)

    # Increase n_estimators and avoid using max_depth for better performance on larger datasets
    rf_model_core = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    rf_model_core.fit(X_train_split, y_train_split)

    # Predictions on validation set for the core set
    y_pred_core = rf_model_core.predict(X_val_split)

    # Evaluation metrics for the core set
    print("Model performance on validation set (Core set):")
    print(confusion_matrix(y_val_split, y_pred_core))
    print(classification_report(y_val_split, y_pred_core))
    print(f"Accuracy: {accuracy_score(y_val_split, y_pred_core)}")

    # Compare the size of the core set with the original PCA-reduced dataset
    print(f"Original PCA-reduced dataset size: {X_pca.shape[0]}")
    print(f"Core set size: {X_core_pca.shape[0]}")
    print(f"Reduction factor: {X_pca.shape[0] / X_core_pca.shape[0]:.2f}")

    # Step 7: Evaluate the core set model on the original PCA-reduced dataset (optional)
    print("Evaluating core set model on the original PCA-reduced dataset...")
    y_pred_original = rf_model_core.predict(X_pca)
    print("Model performance on original PCA-reduced dataset using core set model:")
    print(confusion_matrix(data['Value'], y_pred_original))
    print(classification_report(data['Value'], y_pred_original))
    print(f"Accuracy: {accuracy_score(data['Value'], y_pred_original)}")

if __name__ == "__main__":
    main()
