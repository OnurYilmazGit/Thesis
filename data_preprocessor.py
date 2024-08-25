# Updated data_preprocessor.py to handle the dataset based on the insights

import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA

class DataPreprocessor:
    def __init__(self, data, window_size=10, poly_degree=2):
        self.data = data
        self.window_size = window_size
        self.poly_degree = poly_degree

    def feature_engineering(self):
        features = []
        # Convert the Time column to datetime format
        self.data['Time'] = pd.to_datetime(self.data['Time'], unit='ns')

        # Extracting useful time-based features
        self.data['hour'] = self.data['Time'].dt.hour
        self.data['day'] = self.data['Time'].dt.day
        self.data['month'] = self.data['Time'].dt.month

        # Create rolling mean and standard deviation features for the 'Value' column
        features.append(self.data['Value'].rolling(window=self.window_size).mean().rename('Value_mean'))
        features.append(self.data['Value'].rolling(window=self.window_size).std().rename('Value_std'))

        # Difference feature (rate of change)
        features.append(self.data['Value'].diff().rename('Value_diff'))

        # Concatenate all features at once
        features_df = pd.concat(features, axis=1)
        self.data = pd.concat([self.data, features_df], axis=1)
        self.data.fillna(method='bfill', inplace=True)
        self.data.fillna(method='ffill', inplace=True)

        return self.data

    def normalize_data(self, feature_columns):
        scaler = StandardScaler()
        X = self.data[feature_columns]
        X_scaled = scaler.fit_transform(X)
        return X_scaled

    def reduce_dimensions(self, X_scaled, n_components=20):
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        return X_pca

# The updated data_preprocessor.py class is now ready to be used in the main workflow.
# We have added time-based feature extraction and rolling statistics for the 'Value' column.
