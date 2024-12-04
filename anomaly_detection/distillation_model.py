from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class DistillationModel:
    def __init__(self, base_model=None):
        if base_model is None:
            base_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model = BaggingClassifier(estimator=base_model, n_estimators=10, random_state=42)
    
    def train(self, X_train, y_train):
        # Determine the smallest class size
        min_class_size = y_train.value_counts().min()

        # Check if the dataset is too small for cross-validation
        if min_class_size < 2:
            # Use a train-test split instead of cross-validation
            X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42)
            self.model.fit(X_train_split, y_train_split)
            y_pred = self.model.predict(X_test_split)
            accuracy = accuracy_score(y_test_split, y_pred)
            print(f"Train-Test Split Accuracy: {accuracy:.4f}")
            return accuracy
        else:
            # Perform cross-validation with the adjusted number of folds
            n_folds = min(5, min_class_size)
            scores = cross_val_score(self.model, X_train, y_train, cv=n_folds)
            print(f"Cross-validated accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
            return scores.mean()

    def distill_data(self, X_train, y_train, frac=0.5):  # Increased the fraction here
        distilled_X = X_train.sample(frac=frac, random_state=42)
        distilled_y = y_train.loc[distilled_X.index]
        print(f"Distilled dataset size: {len(distilled_X)}")
        distilled_data = pd.concat([distilled_X, distilled_y], axis=1)
        return distilled_data

    def distill_synthetic_data(self, synthetic_data, frac=0.1):
        distilled_data = synthetic_data.sample(frac=frac, random_state=42)
        print(f"Distilled synthetic dataset size: {len(distilled_data)}")
        return distilled_data
