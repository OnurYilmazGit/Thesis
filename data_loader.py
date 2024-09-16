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

    def load_responses(self):
        response_data_frames = []
        for node in self.nodes:
            response_file = os.path.join(self.responses_path, node, 'responses.csv')
            df = pd.read_csv(response_file)
            df['Node'] = node  # Add a node identifier
            response_data_frames.append(df)
        responses = pd.concat(response_data_frames)
        responses['Time'] = pd.to_datetime(responses['Time'])
        return responses

    def load_sensors(self):
        sensor_data_frames = []
        for node in self.nodes:
            sensor_files = [f for f in os.listdir(os.path.join(self.sensors_path, node)) if f.endswith('.csv')]
            node_data_frames = Parallel(n_jobs=-1)(delayed(pd.read_csv)(os.path.join(self.sensors_path, node, file)) for file in sensor_files)
            node_data_frames = [df.rename(columns={col: file.replace('.csv', '') if col != 'Time' else 'Time' for col in df.columns}) for df, file in zip(node_data_frames, sensor_files)]
            node_sensor_data = pd.concat(node_data_frames, axis=1)
            node_sensor_data = node_sensor_data.loc[:, ~node_sensor_data.columns.duplicated()]
            node_sensor_data['Node'] = node  # Add a node identifier
            sensor_data_frames.append(node_sensor_data)
        sensor_data = pd.concat(sensor_data_frames)
        sensor_data['Time'] = pd.to_datetime(sensor_data['Time'])
        return sensor_data
