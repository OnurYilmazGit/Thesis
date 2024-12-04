import os
import warnings
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer

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
    nodes = [f'node{i}' for i in range(5)]  # Adjusted node count

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

    # Window sizes to test
    window_sizes = [10, 20, 30, 40]
    results = []

    for window_size in window_sizes:
        print(f"\nTesting with window size: {window_size}")
        start_time = time.time()

        # Step 2: Feature engineering and normalization
        print("Performing feature engineering and data normalization...")
        preprocessor = DataPreprocessor(data, window_size=window_size)
        data_preprocessed = preprocessor.feature_engineering()

        # Handle NaN values in the target variable
        data_preprocessed = data_preprocessed.dropna(subset=['Value'])

        # Define the feature columns for normalization (excluding non-feature columns)
        feature_columns = [col for col in data_preprocessed.columns if col not in ['Time', 'Node', 'Value', 'Cluster']]
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
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_pca, data_preprocessed['Value'], test_size=0.2, random_state=42)

        rf_model = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
        rf_model.fit(X_train_split, y_train_split)

        # Predictions on validation set
        y_pred = rf_model.predict(X_val_split)

        # Evaluation metrics
        accuracy = accuracy_score(y_val_split, y_pred)
        print(f"Window Size: {window_size}, Accuracy: {accuracy}")

        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        print(f"Elapsed Time: {elapsed_time:.2f} seconds")

        # Store results
        results.append((window_size, accuracy, elapsed_time))

    # Export results to a text file
    with open("window_size_results.txt", "w") as f:
        for window_size, accuracy, elapsed_time in results:
            f.write(f"Window Size: {window_size}, Accuracy: {accuracy}, Time: {elapsed_time:.2f} seconds\n")

    print("\nAll window sizes have been tested. Results saved to window_size_results.txt")

if __name__ == "__main__":
    main()
