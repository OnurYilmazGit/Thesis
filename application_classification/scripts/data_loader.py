import os
import pandas as pd
from joblib import Parallel, delayed
import warnings

# Suppress FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

class DataLoader:
    def __init__(self, responses_path, sensors_path, nodes):
        self.responses_path = responses_path
        self.sensors_path = sensors_path
        self.nodes = nodes

    def load_responses(self, na_values=[], keep_default_na=False):
        response_data_frames = []
        for node in self.nodes:
            response_file = os.path.join(self.responses_path, node, 'responses.csv')
            df = pd.read_csv(response_file, na_values=na_values, keep_default_na=keep_default_na)
            df['Node'] = node  # Add a node identifier
            response_data_frames.append(df)
        responses = pd.concat(response_data_frames)
        responses['Time'] = pd.to_datetime(responses['Time'])
        return responses

    def load_sensors(self):
        sensor_data_frames = []
        for node in self.nodes:
            sensor_files = [f for f in os.listdir(os.path.join(self.sensors_path, node)) if f.endswith('.csv')]
            node_data_frames = Parallel(n_jobs=-1)(
                delayed(self.read_sensor_file)(os.path.join(self.sensors_path, node, file), file)
                for file in sensor_files
            )
            node_sensor_data = pd.concat(node_data_frames, axis=1)
            node_sensor_data = node_sensor_data.loc[:, ~node_sensor_data.columns.duplicated()]
            node_sensor_data['Node'] = node  # Add a node identifier
            sensor_data_frames.append(node_sensor_data)
        sensor_data = pd.concat(sensor_data_frames)
        sensor_data['Time'] = pd.to_datetime(sensor_data['Time'])
        return sensor_data

    def read_sensor_file(self, file_path, file_name):
        df = pd.read_csv(file_path)
        df.rename(columns={col: file_name.replace('.csv', '') if col != 'Time' else 'Time' for col in df.columns}, inplace=True)
        return df
