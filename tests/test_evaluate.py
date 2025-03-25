# tests/test_evaluate.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import read_config, load_data, split_data
from src.preprocess import preprocess_data
from src.train import save_model
from src.evaluate import evaluate_model
import joblib

def test_evaluation():
    # Load config and data
    config = read_config("src/config/config.yaml")
    df = load_data(config)
    X_train, X_test, y_train, y_test = split_data(df, config)

    # Load preprocessor & model
    preprocessor = joblib.load("models/preprocessor.pkl")
    model = joblib.load("models/best_model.pkl")

    # Preprocess test data
    X_test_processed = preprocessor.transform(X_test)

    # Evaluate model
    metrics = evaluate_model(model, X_test_processed, y_test, report_path="reports/metrics.json")

    print("\n Evaluation complete and metrics saved.")
    return metrics

if __name__ == "__main__":
    test_evaluation()
