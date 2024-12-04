
# main.py

import pandas as pd
import numpy as np
import os
import warnings
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, matthews_corrcoef
from scipy.stats import ks_2samp
from collections import defaultdict
from sklearn.feature_selection import VarianceThreshold

# Import custom classes
from data_loader import DataLoader
from dataPreprocessor import DataPreprocessor

# Suppress warnings
warnings.filterwarnings("ignore")

def evaluate_model(model, X_test, y_test, class_names, model_name, filename_prefix):
    """
    Evaluate the model and generate performance metrics and plots.

    Args:
        model: Trained model to evaluate.
        X_test: Test feature matrix.
        y_test: True labels for the test set.
        class_names: List of class names for plotting.
        model_name: Name of the model (for titles and print statements).
        filename_prefix: Prefix for saved plot filenames.
    Returns:
        metrics_dict: Dictionary containing evaluation metrics.
    """
    print(f"\n=== Evaluating {model_name} ===")
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"{filename_prefix}_confusion_matrix.png")
    plt.close()

    # Classification Report
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    # Compile metrics
    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'cohen_kappa': kappa,
        'mcc': mcc
    }

    return metrics_dict

def compare_class_distributions(y_full, y_core, class_names):
    """
    Compare class distributions between full training set and core set.

    Args:
        y_full: Labels from the full training set.
        y_core: Labels from the core set.
        class_names: List of class names.
    """
    plt.figure(figsize=(10,5))
    sns.countplot(x=y_full, label='Full Training Set', color='blue', alpha=0.6)
    sns.countplot(x=y_core, label='Core Set', color='red', alpha=0.6)
    plt.title('Class Distribution: Full Training Set vs. Core Set')
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig("class_distribution_comparison.png")
    plt.close()

def compare_feature_distributions(X_full, X_core, feature_names):
    """
    Compare feature distributions between full training set and core set using KS test.

    Args:
        X_full: Feature matrix from the full training set.
        X_core: Feature matrix from the core set.
        feature_names: List of feature names.
    """
    p_values = {}
    for i, feature in enumerate(feature_names):
        statistic, p_value = ks_2samp(X_full[:, i], X_core[:, i])
        p_values[feature] = p_value
    # Features with p-value < 0.05
    significant_features = [feature for feature, p in p_values.items() if p < 0.05]
    print(f"Number of features with significantly different distributions: {len(significant_features)}")
    print("Features with significantly different distributions (p < 0.05):")
    print(significant_features)

def plot_pca(X_full, X_core, labels=['Full Training', 'Core Set']):
    """
    Perform PCA and plot the first two principal components.

    Args:
        X_full: Feature matrix from the full training set.
        X_core: Feature matrix from the core set.
        labels: Labels for the datasets.
    """
    from sklearn.decomposition import PCA

    X_combined = np.vstack((X_full, X_core))
    y_combined = np.array([labels[0]] * X_full.shape[0] + [labels[1]] * X_core.shape[0])

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_combined)

    plt.figure(figsize=(10,7))
    sns.scatterplot(x=X_pca[:X_full.shape[0],0], y=X_pca[:X_full.shape[0],1], label=labels[0], alpha=0.3)
    sns.scatterplot(x=X_pca[X_full.shape[0]:,0], y=X_pca[X_full.shape[0]:,1], label=labels[1], alpha=0.7)
    plt.title('PCA Projection of Full Training Set and Core Set')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.savefig("pca_projection.png")
    plt.close()

def main():
    # Initialize variables to store execution times
    execution_times = {}

    # Define class names
    class_names = ['Kripke', 'AMG', 'PENNANT', 'linpack', 'LAMMPS', 'Quicksilver']

    # ================================
    # Step 1: Data Loading and Merging
    # ================================
    start_time = time.time()
    print("\n=== Step 1: Data Loading and Merging ===")

    responses_path = '../responses'
    sensors_path = '../sensors'
    nodes = [f'node{i}' for i in range(2)]
    print(f"Length of Nodes: {len(nodes)}")

    data_loader = DataLoader(responses_path, sensors_path, nodes)
    responses = data_loader.load_responses()
    sensor_data = data_loader.load_sensors()

    # Merge data
    data = pd.merge(sensor_data, responses, on=['Time', 'Node'])

    # Verify loaded nodes
    unique_nodes = data['Node'].unique()
    print("Unique Nodes after Merging:", unique_nodes)
    print("Records per Node:", data['Node'].value_counts())

    # Remove duplicates
    data = data.drop_duplicates()
    print(f"Total records after dropping duplicates: {data.shape[0]}")

    # Map application labels to numerical values
    label_mapping = {
        'Kripke': 0,
        'AMG': 1,
        'PENNANT': 2,
        'linpack': 3,
        'LAMMPS': 4,
        'Quicksilver': 5,
    }

    # Map labels
    data['Value'] = data['Value'].astype(str)  # Ensure labels are strings
    data['Value'] = data['Value'].map(label_mapping)

    # Handle NaN values
    if data['Value'].isnull().any():
        print("Dropping rows with unmapped labels.")
        data.dropna(subset=['Value'], inplace=True)
        data.reset_index(drop=True, inplace=True)

    y = data['Value']
    print("Unique labels after mapping:", y.unique())

    end_time = time.time()
    execution_times['Data Loading and Merging'] = end_time - start_time
    print(f"Data loaded and merged successfully in {execution_times['Data Loading and Merging']:.2f} seconds.")
    print(f"Total records: {data.shape[0]}")

    # Verify no NaN values in y
    if y.isnull().any():
        print("NaN values found in 'y' after handling.")
        return
    else:
        print("No NaN values in 'y' after handling.")

    # ================================
    # Step 2: Time-Based Splitting
    # ================================
    print("\n=== Step 2: Time-Based Splitting ===")
  
    # Sort data by time
    data = data.sort_values('Time').reset_index(drop=True)

    # Determine split index
    split_index = int(len(data) * 0.8)

    # Split data
    data_train = data.iloc[:split_index].reset_index(drop=True)
    data_test = data.iloc[split_index:].reset_index(drop=True)
    print("Training Data Shape:", data_train.shape)
    print("Test Data Shape:", data_test.shape)

    # ================================
    # Step 3: Feature Engineering and Normalization
    # ================================
    print("\n=== Step 3: Feature Engineering and Normalization ===")

    # Apply feature engineering separately
    preprocessor_train = DataPreprocessor(data_train, window_size=20)
    data_train_fe = preprocessor_train.feature_engineering()

    preprocessor_test = DataPreprocessor(data_test, window_size=20)
    data_test_fe = preprocessor_test.feature_engineering()

    # Exclude 'Time' and 'Node' from features
    features_to_exclude = ['Time', 'Node', 'Value']
    # Add any derived features that are based on 'Time' or 'Node' to this list

    # Ensure that the feature columns are the same
    feature_columns_fe = [col for col in data_train_fe.columns if col not in features_to_exclude]
    data_train_fe = data_train_fe[feature_columns_fe + ['Value']]
    data_test_fe = data_test_fe[feature_columns_fe + ['Value']]

    # Update y values after feature engineering and dropping NaNs
    y_train_full = data_train_fe['Value'].reset_index(drop=True)
    y_test_full = data_test_fe['Value'].reset_index(drop=True)

    # Normalize data
    X_train_full = data_train_fe[feature_columns_fe]
    X_test_full = data_test_fe[feature_columns_fe]

    # Handle any remaining NaN values
    X_train_full.fillna(0, inplace=True)
    X_test_full.fillna(0, inplace=True)

    # Fit scaler on training data
    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train_full)

    # Transform test data using the same scaler
    X_test_full_scaled = scaler.transform(X_test_full)

    # Record execution time
    end_time = time.time()
    execution_times['Feature Engineering and Normalization'] = end_time - start_time
    print(f"Feature engineering and normalization completed in {execution_times['Feature Engineering and Normalization']:.2f} seconds.")

    # ================================
    # Step 4: Variance Threshold Feature Selection
    # ================================
    print("\n=== Step 4: Variance Threshold Feature Selection ===")
    start_time = time.time()

    # Initialize the variance threshold selector
    variance_threshold = 0.25  # Adjust as needed
    selector = VarianceThreshold(threshold=variance_threshold)

    # Fit and transform the training data
    X_train_full_var = selector.fit_transform(X_train_full_scaled)
    selected_variance_features = selector.get_support(indices=True)
    selected_variance_feature_names = [feature_columns_fe[i] for i in selected_variance_features]

    # Apply the same transformation to test data
    X_test_full_var = selector.transform(X_test_full_scaled)

    # Calculate the number of features before and after
    print(f"Number of features before Variance Threshold: {X_train_full_scaled.shape[1]}")
    print(f"Number of features after Variance Threshold: {X_train_full_var.shape[1]}")

    # Record execution time
    end_time = time.time()
    execution_times['Variance Threshold Feature Selection'] = end_time - start_time
    print(f"Variance Threshold feature selection completed in {execution_times['Variance Threshold Feature Selection']:.2f} seconds.")

    # ================================
    # Step 5: Entropy-Driven Sampling Strategy
    # ================================
    print("\n=== Step 5: Entropy-Driven Sampling Strategy ===")
    start_time = time.time()


    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

    # Define parameter grid (adjust as needed)
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'criterion': ['gini', 'entropy'],
        'bootstrap': [True, False],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }   

    # Initialize Random Forest
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)

    # Initialize RandomizedSearchCV for full model
    random_search_full = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=50,
        cv=tscv,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )

    # Fit on full training data
    random_search_full.fit(X_train_full_var, y_train_full)

    # Best parameters for full model
    print("Best parameters for full model: ", random_search_full.best_params_)
    print("Best score for full model: ", random_search_full.best_score_)
if __name__ == '__main__':
    main()
