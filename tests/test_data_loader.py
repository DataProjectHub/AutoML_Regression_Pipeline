import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import read_config, load_data, split_data


def test_data_loading():
    config = read_config("src/config/config.yaml")
    df = load_data(config)
    print("Data loaded successfully.")
    print("Shape of dataset:", df.shape)
    print(df.head())

def test_data_split():
    config = read_config("src/config/config.yaml")
    df = load_data(config)
    X_train, X_test, y_train, y_test = split_data(df, config)
    print("\n Data split successfully.")
    print("Training features shape:", X_train.shape)
    print("Test features shape:", X_test.shape)
    print("Training target shape:", y_train.shape)

if __name__ == "__main__":
    test_data_loading()
    test_data_split()
