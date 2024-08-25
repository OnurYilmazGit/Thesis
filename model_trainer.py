from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sklearn.metrics import classification_report, confusion_matrix

class ModelTrainer:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.class_mapping = {label: idx for idx, label in enumerate(['Kripke', 'AMG', 'PENNANT', 'linpack', 'LAMMPS', 'Quicksilver'])}

    def apply_smote(self, sampling_strategy='auto'):
        smote = SMOTE(random_state=42)
        X_synthetic, y_synthetic = smote.fit_resample(self.X_train, self.y_train)
        compression_ratio = len(self.X_train) / len(X_synthetic)
        print(f"Compression Ratio after SMOTE: {compression_ratio:.2f}")
        return X_synthetic, y_synthetic, compression_ratio

    def apply_smote_tomek(self, X_synthetic, y_synthetic, sampling_strategy='auto'):
        smote_tomek = SMOTETomek(random_state=42)
        X_synthetic_tomek, y_synthetic_tomek = smote_tomek.fit_resample(X_synthetic, y_synthetic)
        compression_ratio = len(self.X_train) / len(X_synthetic_tomek)
        print(f"Compression Ratio after SMOTETomek: {compression_ratio:.2f}")
        return X_synthetic_tomek, y_synthetic_tomek, compression_ratio

    def apply_smote_enn(self, X_synthetic, y_synthetic, sampling_strategy='auto'):
        smote_enn = SMOTEENN(random_state=42)
        X_synthetic_enn, y_synthetic_enn = smote_enn.fit_resample(X_synthetic, y_synthetic)
        compression_ratio = len(self.X_train) / len(X_synthetic_enn)
        print(f"Compression Ratio after SMOTEENN: {compression_ratio:.2f}")
        return X_synthetic_enn, y_synthetic_enn, compression_ratio

    def evaluate_model(self, model, model_name, X_test, y_test, synthetic=False):
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(X_test)

        classification_rep = classification_report(y_test, y_pred, output_dict=True, target_names=self.class_mapping.keys())
        conf_matrix = confusion_matrix(y_test, y_pred)

        data_type = "Synthetic" if synthetic else "Real"
        
        # Save the confusion matrix as an image
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_mapping.keys(), yticklabels=self.class_mapping.keys())
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {model_name} ({data_type} Data)')
        image_dir = 'images'
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        plt.savefig(os.path.join(image_dir, f'{model_name}_confusion_matrix_{data_type.lower()}.png'))
        plt.close()

        # Save the classification report as an image
        plt.figure(figsize=(10, 7))
        report_df = pd.DataFrame(classification_rep).iloc[:-1, :].T
        sns.heatmap(report_df, annot=True, cmap='coolwarm')
        plt.title(f'Classification Report - {model_name} ({data_type} Data)')
        plt.savefig(os.path.join(image_dir, f'{model_name}_classification_report_{data_type.lower()}.png'))
        plt.close()

        results = {
            'classification_report': classification_rep,
            'confusion_matrix': conf_matrix
        }
        return results
