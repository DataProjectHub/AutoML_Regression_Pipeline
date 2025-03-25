from src.data_loader import read_config, load_data, split_data
from src.preprocess import preprocess_data
from src.train import train_models, save_model
from src.evaluate import evaluate_model
import joblib
import os

def run_pipeline():
    print("\n AutoML Regression Pipeline Started...")

    # Step 1: Load config and data
    config = read_config("src/config/config.yaml")
    df = load_data(config)
    X_train, X_test, y_train, y_test = split_data(df, config)

    # Step 2: Preprocess data
    X_train_processed, X_test_processed, preprocessor = preprocess_data(X_train, X_test)

    # Step 3: Save preprocessor
    os.makedirs("models", exist_ok=True)
    joblib.dump(preprocessor, "models/preprocessor.pkl")
    print("Preprocessor saved to models/preprocessor.pkl")

    # Step 4: Train and tune models
    best_models = train_models(X_train_processed, y_train)

    # Step 5: Pick best model based on training RMSE
    best_rmse = float("inf")
    best_model = None
    best_model_name = ""
    for name, model in best_models.items():
        preds = model.predict(X_train_processed)
        rmse = ((y_train - preds) ** 2).mean() ** 0.5
        print(f"{name} RMSE on training: {rmse:.4f}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_model_name = name

    # Step 6: Save best model
    save_model(best_model)

    # Step 7: Evaluate on test set
    evaluate_model(best_model, X_test_processed, y_test)

    print(f"\nPipeline complete. Best model: {best_model_name}")

if __name__ == "__main__":
    run_pipeline()
