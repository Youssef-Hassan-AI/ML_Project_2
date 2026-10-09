import os
import joblib
import pandas as pd
from .preprocessing import preprocess_dataframe

_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'saved_models')
_models_cache = {}

def load_model(model_name: str):
    if model_name not in _models_cache:
        model_paths = {
            'XGBoost': 'xgboost_model.pkl',
            'Linear Regression': 'linear_regression_model.pkl',
            'KNN': 'knn_model.pkl',
            'Logistic Regression': 'logistic_regression_model.pkl',
            'SVM': 'svm_model.pkl'
        }
        if model_name in model_paths:
            _models_cache[model_name] = joblib.load(os.path.join(_MODELS_DIR, model_paths[model_name]))
        else:
            raise ValueError(f"Model {model_name} not found.")
    return _models_cache[model_name]

def predict_single(input_data: dict, model_name: str, target_type: str):
    """
    Predict a single instance.
    target_type: 'regression' or 'classification'
    """
    model = load_model(model_name)
    df = pd.DataFrame([input_data])
    X_scaled = preprocess_dataframe(df, target_type=target_type)
    pred = model.predict(X_scaled)[0]
    
    if target_type == 'classification':
        mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
        return mapping.get(pred, "Unknown")
    
    return float(pred)

def predict_batch(df: pd.DataFrame, model_name: str, target_type: str):
    """
    Predict multiple instances.
    """
    model = load_model(model_name)
    X_scaled = preprocess_dataframe(df, target_type=target_type)
    preds = model.predict(X_scaled)
    
    if target_type == 'classification':
        mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
        return [mapping.get(p, "Unknown") for p in preds]
    
    return [float(p) for p in preds]

def get_model_results():
    """
    Return exact verified metrics from the project audit.
    """
    return {
        'regression': [
            {'Model': 'XGBoost', 'R2': 0.9463, 'MAE': 0.9670, 'RMSE': 1.2720, 'Selected': True},
            {'Model': 'KNN', 'R2': 0.9427, 'MAE': 0.4743, 'RMSE': 1.3141, 'Selected': False},
            {'Model': 'Linear Regression', 'R2': 0.8089, 'MAE': 1.9115, 'RMSE': 2.3990, 'Selected': False}
        ],
        'classification': [
            {'Model': 'Logistic Regression', 'Accuracy': 0.7823, 'F1-Score': 0.7820, 'Selected': True},
            {'Model': 'SVM', 'Accuracy': 0.7608, 'F1-Score': 0.7463, 'Selected': False}
        ]
    }
