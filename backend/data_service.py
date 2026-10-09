import pandas as pd
import numpy as np
import os

_DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'my_dataset.csv')
_df_sample_cache = None
_df_stats_cache = None

def get_dataset_sample(n=1000):
    global _df_sample_cache
    if _df_sample_cache is None:
        if os.path.exists(_DATASET_PATH):
            _df_sample_cache = pd.read_csv(_DATASET_PATH).sample(n=n, random_state=42)
        else:
            return pd.DataFrame()
    return _df_sample_cache

def get_dataset_stats():
    global _df_stats_cache
    if _df_stats_cache is not None:
        return _df_stats_cache
        
    if not os.path.exists(_DATASET_PATH):
        return {}
        
    df = get_dataset_sample(n=10000) # Use a 10K sample for stats to avoid full memory load
    stats = {
        'total_rows_approx': 687000, # Approximation from notebook context
        'columns': len(df.columns),
        'missing_values': int(df.isnull().sum().sum()), # based on sample
        'numeric_cols': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_cols': list(df.select_dtypes(include=['object']).columns),
        'summary': df.describe().round(2).reset_index(),
        'feature_names': list(df.columns)
    }
    _df_stats_cache = stats
    return stats

def get_categorical_distributions():
    df = get_dataset_sample(n=5000)
    cat_cols = df.select_dtypes(include=['object']).columns
    dists = {}
    for col in cat_cols:
        dists[col] = df[col].value_counts().reset_index()
        dists[col].columns = [col, 'Count']
    return dists
