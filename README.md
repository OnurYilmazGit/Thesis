# Dataset Compression and Classification in High-Dimensional HPC Environments

This repository contains code and scripts related to **anomaly detection** and **application classification** in high-dimensional, time-series datasets within High-Performance Computing (HPC) environments. The focus aligns with my thesis, which explores dataset compression and classification using techniques such as **entropy-driven sampling**, **K-Means clustering**, **Principal Component Analysis (PCA)**, and **stratified sampling**.

## Table of Contents

- [Folder Structure](#folder-structure)
- [Detailed File Descriptions](#detailed-file-descriptions)
- [Usage Instructions](#usage-instructions)
- [Data Flow and Model Pipeline](#data-flow-and-model-pipeline)
- [Expected Outputs](#expected-outputs)
- [Contribution](#contribution)


## Running the Experiment

Due to the large size of the datasets (+6 GB), they are hosted externally. Please download them from the following links:
- **Anomaly Detection Dataset** and  **Application Classification Dataset**: [Drive Link](https://drive.google.com/drive/u/1/folders/1HKqXq9Ieg44BdsIH8m1stqtOOHEdZM2s)

### Setup Instructions

1. **Install Dependencies**: Before running the scripts, make sure to install the required libraries by running:

   ```bash
   pip install -r requirements.txt

## Folder Structure

```
├── anomaly_detection/
│   ├── sensors/
│   ├── scripts/
│   └── responses/
├── application_classification/
│   ├── sensors/
│   ├── scripts/
│   └── responses/
├── README.md
└── requirements.txt
```

- **anomaly_detection/**: Contains scripts and data specific to anomaly detection tasks.
- **application_classification/**: Contains scripts and data specific to application classification tasks.
- **sensors/**: Stores raw sensor data for training and evaluation.
- **scripts/**: Holds core scripts for classification, sampling, and benchmarking.
- **responses/**: Contains generated outputs for different configurations, including example results of models and benchmarks.


## Detailed File Descriptions

### `benchmarking.py`

- **Purpose**: Evaluates and benchmarks model performance across different sampling and compression methods.
- **Key Functions**:
  - `compare_model_performance`: Compares models on accuracy, F1 score, and other metrics.
  - `statistical_similarity_tests`: Conducts tests like Kolmogorov-Smirnov and Jensen-Shannon for dataset similarity.
  - `feature_importance_comparison`: Examines feature importances across models.
  - `visualize_feature_distributions`: Creates visual representations of feature distributions.
  - `check_data_leakage`: Checks for potential data leakage issues.
- **Usage**: Run after training models to obtain comprehensive performance metrics and dataset similarities.

### `data_loader.py`

- **Purpose**: Manages data loading and preprocessing.
- **Key Functions**:
  - `load_sensor_data`: Loads and preprocesses sensor data.
  - `prepare_datasets`: Splits data into train/test sets with preprocessing.
- **Usage**: Foundational script for consistent data processing across workflows.

### `entropy_classification_with_pca_compression.py`

- **Purpose**: Implements entropy-based classification with PCA for dimensionality reduction.
- **Key Functions**:
  - `train_entropy_based_model`: Trains a model on entropy-sampled, PCA-compressed data.
- **Usage**: Suitable for high-dimensional datasets where entropy-based sampling and PCA enhance efficiency.

### `entropy_classification_with_stratified_sampling.py`

- **Purpose**: Classification using entropy-based sampling and stratified sampling for class distribution.
- **Key Functions**:
  - `stratified_sampling`: Generates core sets with balanced class distribution.
- **Usage**: Ideal for applications requiring balanced sampling without PCA.

### `kmean_classification_with_pca_compression.py`

- **Purpose**: Classification using K-Means clustering and PCA compression.
- **Key Functions**:
  - `kmeans_clustering`: Clusters core data points for K-Means sampling.
- **Usage**: For clustering-based sampling with dimensionality reduction.

### `kmean_classification_with_stratified_sampling.py`

- **Purpose**: Classification using K-Means with stratified sampling, focusing on class distribution.
- **Usage**: For clustering applications that require preserved class distribution.

### `visualization.py`

- **Purpose**: Suite of visualization tools for data analysis.
- **Key Functions**:
  - `plot_scaled_features`, `plot_pca_variance`, `plot_clusters`: Tools for exploring feature distributions, PCA variance, and clusters.
  - `export_for_tsne`: Exports data for t-SNE visualization.
- **Usage**: Generates visualizations supporting interpretability and diagnostics.

## Usage Instructions

### Prerequisites

Ensure you have the necessary dependencies installed. You can install them using:

```bash
pip install -r requirements.txt
```

### Running Scripts

Each configuration corresponds to specific scripts tailored for anomaly detection and application classification. Below are sample commands to execute each configuration:

0. **Dataset Download:** Please download the complete dataset from this [Drive Link](https://drive.google.com/drive/u/1/folders/1HKqXq9Ieg44BdsIH8m1stqtOOHEdZM2s) before running the code.

1. **Entropy Sampling with PCA Compression**

   ```bash
   python anomaly_detection/scripts/entropy_classification_with_pca_compression.py
   ```

2. **Entropy Sampling with Stratified Sampling**

   ```bash
   python anomaly_detection/scripts/entropy_classification_with_stratified_sampling.py
   ```

3. **K-Means Clustering with PCA Compression**

   ```bash
   python application_classification/scripts/kmean_classification_with_pca_compression.py
   ```

4. **K-Means Clustering with Stratified Sampling**

   ```bash
   python application_classification/scripts/kmean_classification_with_stratified_sampling.py
   ```

## Data Flow and Model Pipeline

1. **Data Loading**: Raw sensor data is loaded and preprocessed using `data_loader.py`.
2. **Sampling and Compression**: Depending on the configuration, data is compressed using entropy-based sampling, K-Means clustering, PCA, or stratified sampling.
3. **Model Training**: Compressed data is used to train classification models.
4. **Benchmarking**: `benchmarking.py` evaluates model performance and dataset similarities.
5. **Visualization**: Results are visualized using `visualization.py`.

*Figure 1: Data Flow and Model Pipeline Overview*

## Expected Outputs

For both **anomaly detection** and **application classification**, the repository supports four key configurations, resulting in eight outcomes. Each output provides insights into how different configurations affect performance metrics, feature distributions, and efficiency.

### Configurations

1. **Entropy Sampling with PCA Compression**
2. **Entropy Sampling with Stratified Sampling**
3. **K-Means Clustering with PCA Compression**
4. **K-Means Clustering with Stratified Sampling**

