# Importing necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target
#print(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(y_train)

# Print the training and testing sets
#print("X_train shape:", X_test)

# Train the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
rf_model.fit(X_train, y_train)

# Predict using the trained model
predictions = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Save the trained model
joblib.dump(rf_model, 'trained_rf_model.pkl')


# Return the sample output details
predictions, accuracy, rf_model.feature_importances_, rf_model.get_params()
