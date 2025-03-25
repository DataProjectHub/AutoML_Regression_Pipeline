import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import read_config, load_data, split_data
from src.preprocess import preprocess_data
from src.train import train_models, save_model
import joblib
import numpy as np

def test_training():
    # Load and split data
    config = read_config("src/config/config.yaml")
    df = load_data(config)
    X_train, X_test, y_train, y_test = split_data(df, config)

    # ✅ Preprocess the data
    X_train_processed, X_test_processed, preprocessor = preprocess_data(X_train, X_test)

    # ✅ Save preprocessor
    joblib.dump(preprocessor, 'models/preprocessor.pkl')

    # Train models
    best_models = train_models(X_train_processed, y_train)

    # Compare models on training RMSE
    best_rmse = float("inf")
    best_model = None
    best_model_name = ""

    for name, model in best_models.items():
        preds = model.predict(X_train_processed)
        rmse = np.sqrt(((y_train - preds) ** 2).mean())
        print(f"{name} RMSE on training set: {rmse:.4f}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_model_name = name

    print(f"\nBest model selected: {best_model_name}")
    save_model(best_model)

if __name__ == "__main__":
    test_training()
