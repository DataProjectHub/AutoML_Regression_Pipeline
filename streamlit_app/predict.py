import joblib
import pandas as pd

def load_artifacts(model_path='models/best_model.pkl', preprocessor_path='models/preprocessor.pkl'):
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor

def predict_new_data(new_data_df):
    model, preprocessor = load_artifacts()
    X_processed = preprocessor.transform(new_data_df)
    predictions = model.predict(X_processed)
    return predictions
