from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import joblib

def train_models(X_train, y_train):
    models = {
        "LinearRegression": {
            "model": LinearRegression(),
            "params": {
                "fit_intercept": [True, False]
            }
        },
        "RandomForest": {
            "model": RandomForestRegressor(random_state=42),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5]
            }
        },
        "XGBoost": {
            "model": XGBRegressor(random_state=42, verbosity=0),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1]
            }
        }
    }

    best_models = {}
    for name, mp in models.items():
        print(f"Training and tuning {name}...")
        grid = GridSearchCV(mp["model"], mp["params"], cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_models[name] = grid.best_estimator_
        print(f"Best {name} RMSE: {-grid.best_score_:.4f}")
    
    return best_models

def save_model(model, file_path='models/best_model.pkl'):
    joblib.dump(model, file_path)
    print(f"Model saved to: {file_path}")
