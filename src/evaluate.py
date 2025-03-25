from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import numpy as np
import os

def evaluate_model(model, X_test, y_test, report_path="reports/metrics.json"):
    y_pred = model.predict(X_test)
    
    metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2_Score": r2_score(y_test, y_pred)
    }

    # Save to file
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    return metrics
