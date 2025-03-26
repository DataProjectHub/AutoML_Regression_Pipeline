import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

def read_config(config_path="src/config/config.yaml"):
    """Reads the YAML config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_data(config):
    """Loads the dataset and returns the DataFrame."""
    data_path = config['data_path']
    df = pd.read_csv(data_path)
    return df

def split_data(df, config):
    """Splits data into train/test sets."""
    target = config['target_column']
    X = df.drop(columns=[target])
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['test_size'],
        random_state=config['random_state']
    )
    return X_train, X_test, y_train, y_test
