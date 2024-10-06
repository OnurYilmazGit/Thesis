import pandas as pd
import numpy as np
import time
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from scipy.special import entr
from collections import defaultdict
from sklearn.metrics import classification_report
from benchmarking2 import Benchmarking
from data_loader import DataLoader
from scipy.stats import ks_2samp, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
# Import necessary libraries for validation
# Import necessary libraries for validation
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, chi2_contingency, wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Import necessary libraries for models
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Suppress warnings
warnings.filterwarnings("ignore")

def main():
    # Initialize variables to store execution times
    execution_times = {}

    # Define class names, including 'None' as the 7th label
    class_names = ['Kripke', 'AMG', 'PENNANT', 'linpack', 'LAMMPS', 'Quicksilver', 'None']

    # ================================
    # Step 1: Data Loading and Merging
    # ================================
    start_time = time.time()
    print("\n=== Step 1: Data Loading and Merging ===")

    responses_path = '../responses'
    sensors_path = '../sensors'
    nodes = [f'node{i}' for i in range(16)]
    print(f"Length of Nodes: {len(nodes)}")

    data_loader = DataLoader(responses_path, sensors_path, nodes)
    # Load responses with adjustments to read 'None' as a string
    responses = data_loader.load_responses(na_values=[], keep_default_na=False)
    sensor_data = data_loader.load_sensors()

    # Merge data on 'Time' and 'Node'
    data = pd.merge(sensor_data, responses, on=['Time', 'Node'])

    # Map application labels to numerical values, including 'None' as 6
    label_mapping = {
        'Kripke': 0,
        'AMG': 1,
        'PENNANT': 2,
        'linpack': 3,
        'LAMMPS': 4,
        'Quicksilver': 5,
        'None': 6,
    }

    # Map labels
    data['Value'] = data['Value'].astype(str)  # Ensure labels are strings
    data['Value'] = data['Value'].map(label_mapping)

    # Get labels per 'Time'
    labels = data.groupby('Time')['Value'].first().reset_index()

    # Drop 'Value' from data
    features = data.drop(columns=['Value'])

    # Pivot the data to have one row per 'Time' with node metrics as separate columns
    data_pivot = features.pivot(index='Time', columns='Node')

    # Flatten the multi-level columns
    data_pivot.columns = ['{}_{}'.format(node, feature) for feature, node in data_pivot.columns]

    # Reset index
    data_pivot = data_pivot.reset_index()

    # Merge with labels
    data_final = pd.merge(data_pivot, labels, on='Time')

    # ================================
    # Step 2: Time-Based Splitting
    # ================================
    print("\n=== Step 2: Time-Based Splitting ===")

    # Sort data by time
    data_final = data_final.sort_values('Time').reset_index(drop=True)

    # Determine split index
    split_index = int(len(data_final) * 0.8)

    # Split data
    data_train = data_final.iloc[:split_index].reset_index(drop=True)
    data_test = data_final.iloc[split_index:].reset_index(drop=True)
    print("Training Data Shape:", data_train.shape)
    print("Test Data Shape:", data_test.shape)

    # ================================
    # Step 3: Data Preparation and Normalization
    # ================================
    print("\n=== Step 3: Data Preparation and Normalization ===")

    # Exclude 'Time' from features
    features_to_exclude = ['Time', 'Value']
    # Add any derived features that are based on 'Time' to this list

    feature_columns = [col for col in data_train.columns if col not in features_to_exclude]

    # Separate features and labels
    X_train_full = data_train[feature_columns]
    y_train_full = data_train['Value']

    X_test_full = data_test[feature_columns]
    y_test_full = data_test['Value']

    # Normalize data
    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train_full)
    X_test_full_scaled = scaler.transform(X_test_full)

    # Record execution time
    end_time = time.time()
    execution_times['Data Preparation and Normalization'] = end_time - start_time
    print(f"Data preparation and normalization completed in {execution_times['Data Preparation and Normalization']:.2f} seconds.")

    # ================================
    # Step 4: Variance Threshold Feature Selection
    # ================================
    print("\n=== Step 4: Variance Threshold Feature Selection ===")
    start_time = time.time()

    # Initialize the variance threshold selector
    variance_threshold = 0.1  # Adjust as needed
    selector = VarianceThreshold(threshold=variance_threshold)

    # Fit and transform the training data
    X_train_full_var = selector.fit_transform(X_train_full_scaled)
    selected_variance_features = selector.get_support(indices=True)
    selected_variance_feature_names = [feature_columns[i] for i in selected_variance_features]

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
    # Optimized Entropy-Driven Sampling Strategy with Predictive Uncertainties
    # ================================
    print("\n=== Optimized Entropy-Driven Sampling Strategy ===")
    start_time = time.time()

    # Step 1: Train Random Forest on the full training data with variance-thresholded features
    rf_full = RandomForestClassifier(
        n_estimators=250,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        min_samples_split=10,
        class_weight='balanced_subsample',
        min_samples_leaf=5,
        criterion='entropy',
        bootstrap=True,
    )
    print("Training Random Forest on variance-thresholded full training data...")
    rf_full.fit(X_train_full_var, y_train_full)
    print("Random Forest training completed.")

    # Step 2: Compute Predictive Uncertainties using Entropy
    # Get predicted class probabilities for training data
    probs = rf_full.predict_proba(X_train_full_var)
    print("Predicted class probabilities shape:", probs.shape)

    # Compute entropy for each sample
    predictive_entropies = entr(probs).sum(axis=1)
    print("Predictive entropies shape:", predictive_entropies.shape)

    # Determine thresholds
    high_uncertainty_threshold = np.percentile(predictive_entropies, 99)
    low_uncertainty_threshold = np.percentile(predictive_entropies, 1)

    # Select samples
    high_uncertainty_indices = np.where(predictive_entropies >= high_uncertainty_threshold)[0]
    low_uncertainty_indices = np.where(predictive_entropies <= low_uncertainty_threshold)[0]

    # Combine indices
    selected_indices = np.concatenate([high_uncertainty_indices, low_uncertainty_indices])

    # Step 4: Ensure Balanced Class Distribution
    # Calculate Class Proportions in Full Dataset
    class_distribution_full = y_train_full.value_counts(normalize=True)
    print("\nClass distribution in full dataset (proportion):")
    print(class_distribution_full)

    # Create the Core Dataset
    X_core = X_train_full_var[selected_indices]
    y_core = y_train_full.iloc[selected_indices].reset_index(drop=True)

    # Step 6: Print the Class Distribution in the Core Set
    class_distribution_core = y_core.value_counts(normalize=True)
    print("\nClass distribution in core set (proportion):")
    print(class_distribution_core)

    # Also print counts
    print("\nClass counts in core set:")
    print(y_core.value_counts())

    # Step 7: Check Core Set Size and Compression Ratio
    core_set_size = X_core.nbytes / 1024  # in KB
    reduction_factor = len(X_train_full_var) / len(X_core)

    end_time = time.time()
    execution_times['Optimized Entropy-Driven Sampling'] = end_time - start_time
    print(f"Optimized entropy-driven sampling completed in {execution_times['Optimized Entropy-Driven Sampling']:.2f} seconds.")
    print(f"Core set size: {X_core.shape[0]} samples ({core_set_size:.2f} KB), Reduction factor: {reduction_factor:.2f}")
    print(f"Full set size: {X_train_full_var.nbytes / 1024:.2f} KB")

    # Check class distribution in the core set
    unique, counts = np.unique(y_core, return_counts=True)
    class_counts = dict(zip(unique, counts))
    print("Class distribution in core set:", class_counts)

    # ============================
    # Step 6: Training Random Forest on Core Set Data
    # ============================
    start_time = time.time()
    print("\n=== Step 6: Training Random Forest on Core Set Data ===")

    # Train Random Forest on the core set
    rf_core = RandomForestClassifier(
        n_estimators=150,
        n_jobs=-1,
        class_weight='balanced_subsample',
        bootstrap=True
    )
    print("Training Random Forest on core set...")
    rf_core.fit(X_core, y_core)

    # Evaluate the core model on the test data with variance-thresholded features
    y_pred_core = rf_core.predict(X_test_full_var)

    # Adjust target_names based on actual labels present in y_test_full
    labels = np.unique(y_test_full)
    target_names_adjusted = [class_names[int(i)] for i in labels]

    print("\nClassification Report (Core Set Model on Test Data):")
    print(classification_report(y_test_full, y_pred_core, labels=labels, target_names=target_names_adjusted, zero_division=0))

    # Record execution time
    end_time = time.time()
    execution_times['Training on Core Set'] = end_time - start_time
    print(f"Random Forest trained and evaluated on core set in {execution_times['Training on Core Set']:.2f} seconds.")

    # Calculate compression ratio using data sizes
    compression_ratio_size = X_train_full_var.nbytes / X_core.nbytes
    print(f"Compression Ratio before stratified sampling: {compression_ratio_size:.2f}") 

    # ============================
    # Step 6.5: Further Compression Using Stratified Sampling
    # ============================

    print("\n=== Step 6.5: Further Compression Using Stratified Sampling ===")
    start_time = time.time()

    from sklearn.model_selection import train_test_split

    # Decide on the sampling fraction to achieve the desired compression ratio
    desired_compression_ratio = 20  # Adjust as needed
    sampling_fraction = 1 / desired_compression_ratio  # e.g., 1/50 = 0.02

    print(f"Sampling fraction for desired compression ratio: {sampling_fraction:.4f}")

    # Perform stratified sampling on the core set
    X_core_sampled, _, y_core_sampled, _ = train_test_split(
        X_core, y_core, 
        train_size=sampling_fraction, 
        stratify=y_core, 
        random_state=42
    )

    # Verify the size of the compressed core set
    print(f"Compressed core set size: {X_core_sampled.shape[0]} samples")

    # Check class distribution in compressed core set
    print("Class counts in compressed core set:")
    print(y_core_sampled.value_counts())

    # Record total execution time for this step
    end_time = time.time()
    execution_times['Stratified Sampling Compression'] = end_time - start_time
    print(f"Stratified sampling compression completed in {execution_times['Stratified Sampling Compression']:.2f} seconds.")


    # Calculate compression ratio using data sizes
    compression_ratio_size = X_train_full_var.nbytes / X_core_sampled.nbytes
    print(f"Compression Ratio after stratified sampling: {compression_ratio_size:.2f}") 

    
    # ============================
    # Step 6.6: Training Random Forest on Stratified Compressed Core Set
    # ============================

    print("\n=== Step 6.6: Training Random Forest on Stratified Compressed Core Set ===")
    start_time_rf = time.time()

    rf_core_sampled = RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        class_weight='balanced_subsample',
        bootstrap=True,
        random_state=42
    )
    print("Training Random Forest on stratified compressed core set...")
    rf_core_sampled.fit(X_core_sampled, y_core_sampled)

    # Evaluate the model on the test data
    y_pred_core_sampled = rf_core_sampled.predict(X_test_full_var)

    # Adjust target_names based on actual labels present in y_test_full
    labels = np.unique(y_test_full)
    target_names_adjusted = [class_names[int(i)] for i in labels]

    print("\nClassification Report (Stratified Compressed Core Set Model on Test Data):")
    print(classification_report(y_test_full, y_pred_core_sampled, labels=labels, target_names=target_names_adjusted, zero_division=0))

    # Record execution time
    end_time_rf = time.time()
    execution_times['Training on Stratified Compressed Core Set'] = end_time_rf - start_time_rf
    print(f"Random Forest trained and evaluated on stratified compressed core set in {execution_times['Training on Stratified Compressed Core Set']:.2f} seconds.")

    # =========================
    # Step 7: Comparison of Models
    # =========================
    print("\n=== Step 7: Comparison of Models ===")

    reduction_factor_sampled = len(X_train_full_var) / len(X_core_sampled)

    # Evaluate the original model on the test data with variance-thresholded features
    y_pred_full = rf_full.predict(X_test_full_var)

    # Adjust target_names based on actual labels present in y_test_full
    labels = np.unique(y_test_full)
    target_names_adjusted = [class_names[int(i)] for i in labels]

    print("\nClassification Report (Full Model on Test Data):")
    print(classification_report(y_test_full, y_pred_full, labels=labels, target_names=target_names_adjusted, zero_division=0))

    # Summarize results
    print("\nSummary of Results:")
    print(f"Full Model Test Accuracy: {rf_full.score(X_test_full_var, y_test_full):.4f}")
    print(f"Core Model Test Accuracy: {rf_core.score(X_test_full_var, y_test_full):.4f}")
    print(f"Compressed Core Model Test Accuracy: {rf_core_sampled.score(X_test_full_var, y_test_full):.4f}")
    print(f"Compression Ratio: {reduction_factor:.2f}")
    print(f"Compression Ratio after Stratified Sampling: {reduction_factor_sampled:.2f}")

    # Get core set labels per 'Time' and export it time and labels 
    core_labels = data_final.iloc[selected_indices]

    # Get original dataset labels per 'Time' and export it time and labels
    original_labels = data_final



    # =========================
    # Step 8: Statistical Comparison and Summary
    # =========================
    print("\n=== Step 8: Statistical Comparison and Summary ===")

    # Feature count and data size
    full_data_feature_count = X_train_full_var.shape[1]
    core_data_feature_count = X_core.shape[1]
    sampled_core_data_feature_count = X_core_sampled.shape[1]

    summary_df = pd.DataFrame({
        'Dataset': ['Full Data', 'Core Set', 'Stratified Sampled Core'],
        'Samples': [X_train_full_var.shape[0], X_core.shape[0], X_core_sampled.shape[0]],
        'Accuracy': [
            rf_full.score(X_test_full_var, y_test_full),
            rf_core.score(X_test_full_var, y_test_full),
            rf_core_sampled.score(X_test_full_var, y_test_full)
        ],
        'Compression Ratio': [
            1,
            round(reduction_factor, 2),
            round(reduction_factor_sampled, 2)
        ],
        'Data Size (KB)': [
            X_train_full_var.nbytes / 1024,
            X_core.nbytes / 1024,
            X_core_sampled.nbytes / 1024
        ],
        'Number of Features': [
            full_data_feature_count,
            core_data_feature_count,
            sampled_core_data_feature_count
        ]
    })

    print("\n=== Summary Table ===")
    print(summary_df)

    benchmark = Benchmarking(
        X_full=X_train_full_var,
        y_full=y_train_full,
        X_core=X_core,
        y_core=y_core,
        X_compressed=X_core_sampled,
        y_compressed=y_core_sampled,
        X_test=X_test_full_var,
        y_test=y_test_full,
        feature_names=selected_variance_feature_names,
        class_names=class_names
    )


    # =========================
    # Step 9: Statistical Validation of Compression
    # =========================
    print("\n=== Step 9: Statistical Validation of Compression ===")
    start_time = time.time()

    # 1. Compare feature distributions using Kolmogorov-Smirnov test
    print("\nPerforming Kolmogorov-Smirnov tests on feature distributions...")
    ks_results = []
    for i in range(X_train_full_var.shape[1]):
        stat, p_value = ks_2samp(X_train_full_var[:, i], X_core_sampled[:, i])
        ks_results.append(p_value)
    ks_pvalues = np.array(ks_results)

    # Calculate the percentage of features that have similar distributions
    alpha = 0.05  # Significance level
    num_features = X_train_full_var.shape[1]
    num_similar = np.sum(ks_pvalues > alpha)
    print(f"Number of features with similar distributions: {num_similar}/{num_features}")
    print(f"Percentage: {num_similar / num_features * 100:.2f}%")

    # 3. Compare class distributions using Chi-Square test
    print("\nComparing class distributions using Chi-Square test...")
    full_class_counts = y_train_full.value_counts().sort_index()
    compressed_class_counts = y_core_sampled.value_counts().sort_index()
    contingency_table = pd.DataFrame({
        'Full Data': full_class_counts,
        'Compressed Data': compressed_class_counts
    })
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    print(f"Chi-Square Statistic: {chi2:.2f}, p-value: {p_value:.4f}")
    if p_value > alpha:
        print("Class distributions are similar.")
    else:
        print("Class distributions are significantly different.")
  
    # =========================
    # Step 11: Detailed Feature Similarity Logging
    # =========================
    print("\n=== Step 11: Detailed Feature Similarity Logging ===")
    start_time = time.time()

    # 1. Get Feature Importances from the Full Model
    print("\nCalculating feature importances from the full model...")
    feature_importances = rf_full.feature_importances_

    # Create a DataFrame for feature importances
    feature_importance_df = pd.DataFrame({
        'Feature': selected_variance_feature_names,
        'Importance': feature_importances
    })

    # Sort features by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Select top N important features
    top_n = 50  # You can adjust this number
    top_features = feature_importance_df.head(top_n)['Feature'].tolist()
    top_feature_indices = [selected_variance_feature_names.index(f) for f in top_features]

    # 2. Log Similarity Measures for Top Features
    similarity_logs = []
    print(f"\nLogging similarity measures for top {top_n} important features...")
    for idx in top_feature_indices:
        feature_name = selected_variance_feature_names[idx]
        # Kolmogorov-Smirnov Test
        stat_ks, p_value_ks = ks_2samp(X_train_full_var[:, idx], X_core_sampled[:, idx])
        # Jensen-Shannon Divergence
        hist_full, bin_edges = np.histogram(X_train_full_var[:, idx], bins=50, density=True)
        hist_core, _ = np.histogram(X_core_sampled[:, idx], bins=bin_edges, density=True)
        # Add a small value to avoid zeros
        hist_full += 1e-8
        hist_core += 1e-8
        js_distance = jensenshannon(hist_full, hist_core)
        # Wasserstein Distance
        wasserstein_dist = wasserstein_distance(X_train_full_var[:, idx], X_core_sampled[:, idx])
        # Append to logs
        similarity_logs.append({
            'Feature': feature_name,
            'KS Statistic': stat_ks,
            'KS p-value': p_value_ks,
            'Jensen-Shannon Distance': js_distance,
            'Wasserstein Distance': wasserstein_dist
        })

    # Convert logs to DataFrame
    similarity_logs_df = pd.DataFrame(similarity_logs)

    # Save logs to a CSV file
    similarity_logs_df.to_csv('feature_similarity_logs.csv', index=False)
    print("\nFeature similarity logs saved to 'feature_similarity_logs.csv'.")

    # 3. Display the logs
    print("\nTop Features Similarity Measures:")
    print(similarity_logs_df.head(10))  # Display top 10 for brevity

    # 4. Visualize Distributions of Top Features
    print("\nVisualizing distributions of top important features...")
    for idx in top_feature_indices[:5]:  # Visualize top 5 features
        feature_name = selected_variance_feature_names[idx]
        plt.figure(figsize=(8, 4))
        sns.kdeplot(X_train_full_var[:, idx], label='Full Data', shade=True)
        sns.kdeplot(X_core_sampled[:, idx], label='Compressed Data', shade=True)
        plt.title(f'Distribution Comparison for Feature: {feature_name}')
        plt.legend()
        plt.show()

    # Record execution time
    end_time = time.time()
    execution_times['Feature Similarity Logging'] = end_time - start_time
    print(f"Feature similarity logging completed in {execution_times['Feature Similarity Logging']:.2f} seconds.")
    #benchmark.cross_validation_checks(rf_core)



if __name__ == '__main__':
    main()
