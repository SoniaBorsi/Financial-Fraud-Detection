import pandas as pd
import yaml
import os

def load_data(config_path="config/config.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    dataset_path = os.path.join('datasets', config['dataset'])
    data = pd.read_csv(dataset_path)

    print(f"Data loaded successfully: {dataset_path}, Shape: {data.shape}")
    return data