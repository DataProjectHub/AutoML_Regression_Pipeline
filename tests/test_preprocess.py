import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import read_config, load_data, split_data
from src.preprocess import preprocess_data

def test_preprocessing():
    # Load config and data
    config = read_config("src/config/config.yaml")
    df = load_data(config)
    X_train, X_test, y_train, y_test = split_data(df, config)
    
    # Apply preprocessing
    X_train_processed, X_test_processed, preprocessor = preprocess_data(X_train, X_test)
    
    print("Preprocessing complete.")
    print("Processed training features shape:", X_train_processed.shape)
    print("Processed test features shape:", X_test_processed.shape)
    print("Preprocessor object:", preprocessor)

if __name__ == "__main__":
    test_preprocessing()
