import os
import pandas as pd
import numpy as np
import joblib

_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'saved_models')

_le_gender = None
_feature_names = None
_scaler_reg = None
_scaler_clf = None

def load_preprocessors():
    global _le_gender, _feature_names, _scaler_reg, _scaler_clf
    if _le_gender is None:
        _le_gender = joblib.load(os.path.join(_MODELS_DIR, 'label_encoder_gender.pkl'))
        _feature_names = joblib.load(os.path.join(_MODELS_DIR, 'feature_names.pkl'))
        _scaler_reg = joblib.load(os.path.join(_MODELS_DIR, 'scaler_regression.pkl'))
        _scaler_clf = joblib.load(os.path.join(_MODELS_DIR, 'scaler_classification.pkl'))

def preprocess_dataframe(df: pd.DataFrame, target_type='regression'):
    """
    Applies exact preprocessing steps used in training.
    target_type: 'regression' or 'classification'
    """
    load_preprocessors()
    df_proc = df.copy()
    
    # Drop ID if present
    if 'ID' in df_proc.columns:
        df_proc.drop(columns=['ID'], inplace=True)
        
    # Fill missing health_condition
    if 'health_condition' in df_proc.columns:
        df_proc['health_condition'].fillna('None', inplace=True)
        
    # Categorical handling
    if 'gender' in df_proc.columns:
        # Handle unseen labels by falling back to 0 or mode (here let's just transform known)
        try:
            df_proc['gender'] = _le_gender.transform(df_proc['gender'])
        except ValueError:
            # If unseen gender passed, just map to 0 (default)
            mapping = {cl: i for i, cl in enumerate(_le_gender.classes_)}
            df_proc['gender'] = df_proc['gender'].map(mapping).fillna(0).astype(int)

    if 'intensity' in df_proc.columns:
        intensity_order = {'Low': 0, 'Medium': 1, 'High': 2}
        df_proc['intensity'] = df_proc['intensity'].map(intensity_order).fillna(1) # fallback Medium
        
    # One-Hot Encoding
    cols_to_encode = [c for c in ['activity_type', 'smoking_status', 'health_condition'] if c in df_proc.columns]
    df_proc = pd.get_dummies(df_proc, columns=cols_to_encode, drop_first=False)
    
    # Instead of drop_first in live inference, we just ensure all expected columns exist
    for col in _feature_names:
        if col not in df_proc.columns:
            df_proc[col] = 0
            
    # Keep only expected columns in order
    X = df_proc[_feature_names].copy()
    
    # Ensure types are numeric
    X = X.astype(float)
    
    if target_type == 'regression':
        X_scaled = _scaler_reg.transform(X)
    else:
        X_scaled = _scaler_clf.transform(X)
        
    return X_scaled

def preprocess_single(input_data: dict, target_type='regression'):
    """
    input_data is a dict from Gradio inputs.
    """
    df = pd.DataFrame([input_data])
    return preprocess_dataframe(df, target_type)
