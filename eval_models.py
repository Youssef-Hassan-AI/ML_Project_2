import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('my_dataset.csv').sample(n=10000, random_state=42)

# Preprocessing exactly like the notebook
df.drop(columns=['ID'], inplace=True, errors='ignore')
df['health_condition'].fillna('None', inplace=True)

le_gender = joblib.load('saved_models/label_encoder_gender.pkl')
df['gender'] = le_gender.transform(df['gender'])

intensity_order = {'Low': 0, 'Medium': 1, 'High': 2}
df['intensity'] = df['intensity'].map(intensity_order)

df = pd.get_dummies(df, columns=['activity_type', 'smoking_status', 'health_condition'], drop_first=True, dtype=int)

# Target for regression
y_reg = df['fitness_level']

# Target for classification
bins = [df['fitness_level'].min() - 0.01, 6.87, 13.74, df['fitness_level'].max() + 0.01]
labels = [0, 1, 2]
df['fitness_category'] = pd.cut(df['fitness_level'], bins=bins, labels=labels).astype(int)
y_clf = df['fitness_category']

# Features
X = df.drop(columns=['fitness_level', 'fitness_category'], errors='ignore')
# Ensure column order matches
expected_cols = joblib.load('saved_models/feature_names.pkl')
for col in expected_cols:
    if col not in X.columns:
        X[col] = 0
X = X[expected_cols]

# Scale
scaler_reg = joblib.load('saved_models/scaler_regression.pkl')
X_reg_scaled = scaler_reg.transform(X)

scaler_clf = joblib.load('saved_models/scaler_classification.pkl')
X_clf_scaled = scaler_clf.transform(X)

# Load models
xgb = joblib.load('saved_models/xgboost_model.pkl')
lr = joblib.load('saved_models/linear_regression_model.pkl')
knn = joblib.load('saved_models/knn_model.pkl')
log = joblib.load('saved_models/logistic_regression_model.pkl')
svm = joblib.load('saved_models/svm_model.pkl')

print("=== REGRESSION ===")
for name, model in [('XGBoost', xgb), ('Linear Regression', lr), ('KNN', knn)]:
    y_pred = model.predict(X_reg_scaled)
    r2 = r2_score(y_reg, y_pred)
    mae = mean_absolute_error(y_reg, y_pred)
    rmse = np.sqrt(mean_squared_error(y_reg, y_pred))
    print(f"{name}: R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

print("=== CLASSIFICATION ===")
for name, model in [('Logistic Regression', log), ('SVM', svm)]:
    y_pred = model.predict(X_clf_scaled)
    acc = accuracy_score(y_clf, y_pred)
    f1 = f1_score(y_clf, y_pred, average='weighted')
    print(f"{name}: Acc={acc:.4f}, F1={f1:.4f}")

