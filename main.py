import os
import warnings
import pandas as pd
import time  # Time module for timing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from fpdf import FPDF

# Import custom classes
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from model_trainer import ModelTrainer
from report_generator import ReportGenerator

from classification_models.knn import get_model as get_knn
from classification_models.random_forest import get_model as get_random_forest
from classification_models.logistic_regression import get_model as get_logistic_regression
from classification_models.lightgbm import get_model as get_lightgbm
from classification_models.decision_tree import get_model as get_decision_tree

# Suppress FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# Define paths
responses_path = '../responses'
sensors_path = '../sensors'
nodes = [f'node{i}' for i in range(4)]

# Step 1: Load and preprocess data
print("Loading response and sensor data...")
data_loader = DataLoader(responses_path, sensors_path, nodes)
responses = data_loader.load_responses()
print("Responses data loaded successfully.")
sensor_data = data_loader.load_sensors()
print("Sensor data loaded successfully.")

print("Merging response and sensor data...")
data = pd.merge(sensor_data, responses, on=['Time', 'Node'])
data['Value'] = data['Value'].map({label: idx for idx, label in enumerate(['Kripke', 'AMG', 'PENNANT', 'linpack', 'LAMMPS', 'Quicksilver'])})
print("Data merged successfully.")

# Step 2: Feature engineering and normalization
print("Performing feature engineering and data normalization...")
preprocessor = DataPreprocessor(data)
data = preprocessor.feature_engineering()
print("Feature engineering completed.")
X_scaled = preprocessor.normalize_data([col for col in data.columns if col not in ['Time', 'Node', 'Value']])
X_pca = preprocessor.reduce_dimensions(X_scaled)
print("Data normalization and dimensionality reduction completed.")

# Step 3: Train and evaluate models
print("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_pca, data['Value'], test_size=0.2, random_state=42)
print("Data split completed.")

# Now you can use these imported functions to instantiate your models
models = {
    #'k-NN': get_knn()
    'Random Forest': get_random_forest()
    #'Logistic Regression': get_logistic_regression(),
    #'LightGBM': get_lightgbm(),
    #'Decision Tree': get_decision_tree()
}

trainer = ModelTrainer(X_train, y_train)

# Train and evaluate models on the real data
results_real = {}
for model_name, model in models.items():
    print(f"Training and evaluating model: {model_name}...")
    results_real[model_name] = trainer.evaluate_model(model, model_name, X_test, y_test)
    print(f"Model {model_name} evaluated successfully.")

    # Generate reports
    print(f"Generating report for model: {model_name}...")
    image_dir = 'images'
    save_dir = 'reports'
    report_generator = ReportGenerator(results_real[model_name], model_name)
    pdf_path = report_generator.generate_pdf(image_dir, save_dir)
    print(f"Report for {model_name} saved as '{pdf_path}'")

print("Initial real-data evaluation completed. Review the results before applying SMote or other enhancements.")

# Step 4: Apply SMOTE and other data augmentation techniques
print("Applying SMOTE and other data augmentation techniques...")

# Create synthetic training and testing data using SMOTE
X_synthetic_train, y_synthetic_train, smote_train_ratio = trainer.apply_smote(sampling_strategy='auto')
X_synthetic_test, y_synthetic_test, smote_test_ratio = trainer.apply_smote(sampling_strategy='auto')

# Apply SMOTETomek and SMOTEENN on the training data
X_synthetic_tomek_train, y_synthetic_tomek_train, tomek_train_ratio = trainer.apply_smote_tomek(X_synthetic_train, y_synthetic_train, sampling_strategy='auto')
X_synthetic_enn_train, y_synthetic_enn_train, enn_train_ratio = trainer.apply_smote_enn(X_synthetic_train, y_synthetic_train, sampling_strategy='auto')

results_synthetic = {}

for model_name, model in models.items():
    print(f"Training and evaluating model: {model_name} on synthetic data...")
    
    # Train the model on synthetic data
    model.fit(X_synthetic_train, y_synthetic_train)
    
    # Evaluate the model on the synthetic test set
    y_pred_synthetic = model.predict(X_synthetic_test)
    
    # Generate classification report and confusion matrix
    classification_rep = classification_report(y_synthetic_test, y_pred_synthetic, output_dict=True, target_names=trainer.class_mapping.keys())
    conf_matrix = confusion_matrix(y_synthetic_test, y_pred_synthetic)

    # Save results
    results_synthetic[model_name] = {
        'classification_report': classification_rep,
        'confusion_matrix': conf_matrix
    }

    print(f"Model {model_name} on synthetic data evaluated successfully.")

    # Generate reports for synthetic data
    print(f"Generating synthetic data report for model: {model_name}...")
    report_generator = ReportGenerator(results_synthetic[model_name], model_name, data_type='synthetic', compression_ratio=smote_train_ratio)
    pdf_path = report_generator.generate_pdf(image_dir, save_dir)
    print(f"Synthetic data report for {model_name} saved as '{pdf_path}'")

print("Process completed successfully.")
