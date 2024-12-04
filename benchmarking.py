# benchmarking.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, cohen_kappa_score,
                             matthews_corrcoef)
from sklearn.model_selection import cross_val_score, learning_curve
import numpy as np
import multiprocessing


def plot_confusion_matrix(conf_matrix, class_names, title, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_learning_curve(estimator, X, y, title="Learning Curve", cv=5, n_jobs=-1, scoring=None, train_sizes=np.linspace(0.1, 1.0, 10)):
    """
    Plots the learning curve for a given estimator.
    
    Args:
        estimator: The machine learning model to evaluate.
        X: Feature matrix.
        y: Target vector.
        title: Title of the plot.
        cv: Number of cross-validation splits.
        n_jobs: Number of jobs to run in parallel (-1 uses all available cores).
        scoring: Scoring metric to use. Default is None (use estimator's default).
        train_sizes: Array of training sizes to use for generating the learning curve.
    """
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    # Compute learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=scoring)

    # Calculate mean and standard deviation for training and validation scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    plt.grid()
    plt.tight_layout()
    plt.show()


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
import numpy as np

def is_scaled(X):
    """Check if the data is scaled (mean close to 0 and standard deviation close to 1)."""
    mean_check = np.allclose(np.mean(X, axis=0), 0, atol=1e-2)  # Mean close to 0
    std_check = np.allclose(np.std(X, axis=0), 1, atol=1e-2)    # Std close to 1
    return mean_check and std_check

def lasso_feature_selection(X, y, alpha=0.001):
    """
    Perform feature selection using Lasso (L1 Regularization), utilizing full CPU.
    
    Args:
        X: Feature matrix.
        y: Target vector.
        alpha: Regularization strength (the higher the alpha, the more features are eliminated).
        
    Returns:
        X_selected: The reduced feature matrix after Lasso.
        selected_features: Indices of the selected features.
    """
    print("\n=== Performing Lasso Feature Selection (L1 Regularization) ===")

    # Check if X is already scaled
    if is_scaled(X):
        print("Data is already scaled.")
        X_scaled = X
    else:
        print("Data is not scaled. Applying StandardScaler...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

    # Apply Lasso for feature selection (with multi-threading)
    lasso = Lasso(alpha=alpha, max_iter=400)
    lasso.fit(X_scaled, y)
    
    # Select features with non-zero coefficients
    selected_features = np.where(lasso.coef_ != 0)[0]
    
    # Reduce the feature matrix
    X_selected = X[:, selected_features]
    
    print(f"Selected {len(selected_features)} features out of {X.shape[1]} using Lasso.")
    
    return X_selected, selected_features


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
    plot_confusion_matrix(conf_matrix, class_names, f"Confusion Matrix - {model_name}",
                          f"{filename_prefix}_confusion_matrix.png")

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

def perform_cross_validation(model, X, y, cv=5):
    """
    Perform cross-validation and return the scores.

    Args:
        model: The machine learning model to evaluate.
        X: Feature matrix.
        y: Target vector.
        cv: Number of cross-validation folds.
    Returns:
        cv_scores: Array of cross-validation scores.
    """
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    return cv_scores

def statistical_comparison(cv_scores_full, cv_scores_core):
    """
    Perform statistical comparison between two sets of cross-validation scores.

    Args:
        cv_scores_full: Cross-validation scores from the full model.
        cv_scores_core: Cross-validation scores from the core model.
    """
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(cv_scores_full, cv_scores_core, equal_var=False)
    print("\n=== Statistical Comparison of Models ===")
    print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")
    if p_value > 0.05:
        print("No significant difference between the full data model and the core data model.")
    else:
        print("Significant difference between the full data model and the core data model.")
