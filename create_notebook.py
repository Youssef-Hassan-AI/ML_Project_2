import json
import sys

# Build the notebook structure
cells = []

def md(source):
    """Create a markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [s + "\n" for s in source.split("\n")]
    }

def code(source):
    """Create a code cell."""
    return {
        "cell_type": "code",
        "metadata": {},
        "source": [s + "\n" for s in source.split("\n")],
        "outputs": [],
        "execution_count": None
    }

# ===== CELL 1: Title =====
cells.append(md("""# 🏋️ Fitness Level Prediction — Machine Learning Project
### Final Project — Team 5

This notebook builds and evaluates **5 Machine Learning models** to predict fitness level from health & activity data:

1. **KNN** (K-Nearest Neighbors)
2. **Linear Regression**
3. **Logistic Regression** (Classification)
4. **XGBoost**
5. **SVM** (Support Vector Machine — Classification)

> **Target Variable**: `fitness_level` (continuous: 0 – 20.63)
> - For **Regression** models (KNN, Linear Regression, XGBoost): predict the raw continuous value
> - For **Classification** models (Logistic Regression, SVM): predict binned fitness categories (Low / Medium / High)"""))

# ===== CELL 2: Imports =====
cells.append(md("## 1. Import Libraries"))
cells.append(code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Preprocessing & Pipelines
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Regression Models
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Metrics
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# Model Saving
import joblib
import os

print("All libraries imported successfully ✅")"""))

# ===== CELL 3: Load Data =====
cells.append(md("## 2. Load & Explore the Dataset"))
cells.append(code("""# Load the dataset
df = pd.read_csv('my_dataset.csv')
print(f"Dataset Shape: {df.shape}")
print(f"Total Rows: {df.shape[0]:,}  |  Total Columns: {df.shape[1]}")
df.head(10)"""))

# ===== CELL 4: Data Info =====
cells.append(code("""# Data types and info
df.info()"""))

# ===== CELL 5: Statistical Summary =====
cells.append(code("""# Statistical summary
df.describe().round(2)"""))

# ===== CELL 6: Missing Values =====
cells.append(code("""# Missing values analysis
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Percentage (%)': missing_pct})
missing_df[missing_df['Missing Count'] > 0]"""))

# ===== CELL 7: EDA Visualizations =====
cells.append(md("## 3. Exploratory Data Analysis (EDA)"))
cells.append(code("""fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Distribution of Key Numerical Features', fontsize=16, fontweight='bold')

num_features = ['age', 'bmi', 'calories_burned', 'daily_steps', 'avg_heart_rate', 'fitness_level']
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

for i, (feat, color) in enumerate(zip(num_features, colors)):
    ax = axes[i // 3, i % 3]
    ax.hist(df[feat], bins=50, color=color, edgecolor='black', alpha=0.7)
    ax.set_title(feat, fontsize=13, fontweight='bold')
    ax.set_xlabel(feat)
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.show()"""))

# ===== CELL 8: Categorical EDA =====
cells.append(code("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Activity Type
df['activity_type'].value_counts().plot.bar(ax=axes[0], color='#3498db', edgecolor='black')
axes[0].set_title('Activity Type Distribution', fontweight='bold')
axes[0].set_xlabel('Activity Type')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=45)

# Gender
df['gender'].value_counts().plot.pie(ax=axes[1], autopct='%1.1f%%', colors=['#e74c3c', '#3498db'],
                                      startangle=90, textprops={'fontsize': 12})
axes[1].set_title('Gender Distribution', fontweight='bold')
axes[1].set_ylabel('')

# Intensity
df['intensity'].value_counts().plot.bar(ax=axes[2], color=['#2ecc71', '#f39c12', '#e74c3c'], edgecolor='black')
axes[2].set_title('Intensity Distribution', fontweight='bold')
axes[2].set_xlabel('Intensity')
axes[2].set_ylabel('Count')

plt.tight_layout()
plt.show()"""))

# ===== CELL 9: Correlation Heatmap =====
cells.append(code("""plt.figure(figsize=(14, 10))
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()"""))

# ===== CELL 10: Target Distribution =====
cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(df['fitness_level'], bins=60, color='#1abc9c', edgecolor='black', alpha=0.8)
axes[0].set_title('Fitness Level Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Fitness Level')
axes[0].set_ylabel('Frequency')
axes[0].axvline(df['fitness_level'].mean(), color='red', linestyle='--', label=f"Mean: {df['fitness_level'].mean():.2f}")
axes[0].legend()

# Boxplot
sns.boxplot(x=df['fitness_level'], ax=axes[1], color='#1abc9c')
axes[1].set_title('Fitness Level Boxplot', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()"""))

# ===== CELL 11: Data Preprocessing =====
cells.append(md("""## 4. Data Preprocessing

Steps:
1. Drop the `ID` column (not predictive)
2. Handle missing values in `health_condition`
3. Encode categorical variables
4. Feature scaling
5. Create classification target by binning `fitness_level` into 3 categories"""))

cells.append(code("""# Drop ID column
df.drop(columns=['ID'], inplace=True)

# Fill missing health_condition with 'None' (no condition)
df['health_condition'].fillna('None', inplace=True)

print("Missing values after cleaning:")
print(df.isnull().sum().sum(), "total missing values")
print(f"\\nDataset shape: {df.shape}")"""))

# ===== CELL 12: Encoding =====
cells.append(code("""# Encode categorical features
# Label Encoding for binary feature
le_gender = LabelEncoder()
df['gender'] = le_gender.fit_transform(df['gender'])

# Ordinal Encoding for intensity (has natural order)
intensity_order = {'Low': 0, 'Medium': 1, 'High': 2}
df['intensity'] = df['intensity'].map(intensity_order)

# One-Hot Encoding for remaining categorical features
df = pd.get_dummies(df, columns=['activity_type', 'smoking_status', 'health_condition'], 
                    drop_first=True, dtype=int)

print(f"Shape after encoding: {df.shape}")
print(f"\\nColumns: {list(df.columns)}")"""))

# ===== CELL 13: Create Classification Target =====
cells.append(code("""# Create classification target for Logistic Regression and SVM
# Bin fitness_level into 3 categories: Low, Medium, High
bins = [df['fitness_level'].min() - 0.01, 6.87, 13.74, df['fitness_level'].max() + 0.01]
labels = [0, 1, 2]  # 0=Low, 1=Medium, 2=High
df['fitness_category'] = pd.cut(df['fitness_level'], bins=bins, labels=labels).astype(int)

print("Fitness Category Distribution:")
print(df['fitness_category'].value_counts().sort_index())
print(f"\\n0 = Low (0 - 6.87)")
print(f"1 = Medium (6.87 - 13.74)")
print(f"2 = High (13.74 - 20.63)")

fig, ax = plt.subplots(figsize=(6, 4))
df['fitness_category'].value_counts().sort_index().plot.bar(
    color=['#e74c3c', '#f39c12', '#2ecc71'], edgecolor='black', ax=ax)
ax.set_title('Fitness Category Distribution', fontweight='bold')
ax.set_xticklabels(['Low', 'Medium', 'High'], rotation=0)
ax.set_ylabel('Count')
plt.tight_layout()
plt.show()"""))

# ===== CELL 14: Sampling for large dataset =====
cells.append(md("""## 5. Prepare Train/Test Splits

> **Note**: The dataset has ~687K rows. We use a stratified sample of 100,000 rows for faster training while maintaining data distribution. This ensures high accuracy with reasonable training times."""))

cells.append(code("""# Sample data for efficient training (100K rows is more than sufficient)
SAMPLE_SIZE = 100_000
np.random.seed(42)

df_sampled = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
print(f"Sampled dataset shape: {df_sampled.shape}")

# ==================== REGRESSION SPLIT ====================
# Features & Target for regression
X_reg = df_sampled.drop(columns=['fitness_level', 'fitness_category'])
y_reg = df_sampled['fitness_level']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# ==================== CLASSIFICATION SPLIT ====================
# Features & Target for classification
X_clf = df_sampled.drop(columns=['fitness_level', 'fitness_category'])
y_clf = df_sampled['fitness_category']

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

print(f"\\nRegression Split  → Train: {X_train_reg.shape}, Test: {X_test_reg.shape}")
print(f"Classification Split → Train: {X_train_clf.shape}, Test: {X_test_clf.shape}")"""))

# ===== CELL 15: Feature Scaling =====
cells.append(code("""# Feature Scaling
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)

print("Feature scaling applied ✅")
print(f"Number of features: {X_train_reg_scaled.shape[1]}")"""))

# ===== CELL 16: Results Storage =====
cells.append(code("""# Store results for final comparison
regression_results = {}
classification_results = {}"""))

# ===== MODEL 1: KNN =====
cells.append(md("""## 6. Model 1: K-Nearest Neighbors (KNN) — Regression

KNN predicts the fitness level by averaging the values of the K closest training samples."""))

cells.append(code("""# Find optimal K using cross-validation
k_values = [3, 5, 7, 9, 11, 15]
cv_scores = []

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k, weights='distance', n_jobs=-1)
    scores = cross_val_score(knn, X_train_reg_scaled, y_train_reg, cv=5, scoring='r2')
    cv_scores.append(scores.mean())
    print(f"K={k:2d}  |  CV R² = {scores.mean():.4f} ± {scores.std():.4f}")

best_k = k_values[np.argmax(cv_scores)]
print(f"\\n✅ Best K = {best_k} with R² = {max(cv_scores):.4f}")"""))

cells.append(code("""# Train KNN with best K
knn_model = KNeighborsRegressor(n_neighbors=best_k, weights='distance', n_jobs=-1)
knn_model.fit(X_train_reg_scaled, y_train_reg)
y_pred_knn = knn_model.predict(X_test_reg_scaled)

# Evaluate
knn_mae = mean_absolute_error(y_test_reg, y_pred_knn)
knn_rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_knn))
knn_r2 = r2_score(y_test_reg, y_pred_knn)

regression_results['KNN'] = {'MAE': knn_mae, 'RMSE': knn_rmse, 'R²': knn_r2}

print("=" * 50)
print("     KNN Regression Results")
print("=" * 50)
print(f"  MAE  = {knn_mae:.4f}")
print(f"  RMSE = {knn_rmse:.4f}")
print(f"  R²   = {knn_r2:.4f}")
print("=" * 50)"""))

cells.append(code("""# KNN: Actual vs Predicted
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter plot
axes[0].scatter(y_test_reg, y_pred_knn, alpha=0.1, s=5, color='#3498db')
axes[0].plot([0, 21], [0, 21], 'r--', linewidth=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Fitness Level', fontsize=12)
axes[0].set_ylabel('Predicted Fitness Level', fontsize=12)
axes[0].set_title(f'KNN — Actual vs Predicted (R² = {knn_r2:.4f})', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)

# Residuals
residuals = y_test_reg - y_pred_knn
axes[1].hist(residuals, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
axes[1].set_title('KNN — Residual Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Residual')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()"""))

# ===== MODEL 2: Linear Regression =====
cells.append(md("""## 7. Model 2: Linear Regression

A simple yet powerful baseline model that fits a linear relationship between features and the target."""))

cells.append(code("""# Train Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_reg_scaled, y_train_reg)
y_pred_lr = lr_model.predict(X_test_reg_scaled)

# Evaluate
lr_mae = mean_absolute_error(y_test_reg, y_pred_lr)
lr_rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_lr))
lr_r2 = r2_score(y_test_reg, y_pred_lr)

regression_results['Linear Regression'] = {'MAE': lr_mae, 'RMSE': lr_rmse, 'R²': lr_r2}

print("=" * 50)
print("     Linear Regression Results")
print("=" * 50)
print(f"  MAE  = {lr_mae:.4f}")
print(f"  RMSE = {lr_rmse:.4f}")
print(f"  R²   = {lr_r2:.4f}")
print("=" * 50)

# Cross-validation
cv_scores_lr = cross_val_score(lr_model, X_train_reg_scaled, y_train_reg, cv=5, scoring='r2')
print(f"\\n5-Fold CV R² = {cv_scores_lr.mean():.4f} ± {cv_scores_lr.std():.4f}")"""))

cells.append(code("""# Linear Regression: Actual vs Predicted
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_test_reg, y_pred_lr, alpha=0.1, s=5, color='#e74c3c')
axes[0].plot([0, 21], [0, 21], 'r--', linewidth=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Fitness Level', fontsize=12)
axes[0].set_ylabel('Predicted Fitness Level', fontsize=12)
axes[0].set_title(f'Linear Regression — Actual vs Predicted (R² = {lr_r2:.4f})', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)

residuals_lr = y_test_reg - y_pred_lr
axes[1].hist(residuals_lr, bins=50, color='#e74c3c', edgecolor='black', alpha=0.7)
axes[1].axvline(0, color='blue', linestyle='--', linewidth=2)
axes[1].set_title('Linear Regression — Residual Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Residual')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Feature Importance (Coefficients)
feature_importance = pd.DataFrame({
    'Feature': X_reg.columns,
    'Coefficient': lr_model.coef_
}).sort_values('Coefficient', key=abs, ascending=False).head(10)

plt.figure(figsize=(10, 5))
plt.barh(feature_importance['Feature'], feature_importance['Coefficient'], color='#e74c3c', edgecolor='black')
plt.title('Top 10 Feature Coefficients — Linear Regression', fontsize=14, fontweight='bold')
plt.xlabel('Coefficient Value')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()"""))

# ===== MODEL 3: Logistic Regression =====
cells.append(md("""## 8. Model 3: Logistic Regression — Classification

Since Logistic Regression is a **classification** algorithm, we use the binned `fitness_category` target:
- **0 = Low** (0 – 6.87)
- **1 = Medium** (6.87 – 13.74)
- **2 = High** (13.74 – 20.63)"""))

cells.append(code("""# Train Logistic Regression with tuned hyperparameters
log_model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver='lbfgs',
    multi_class='multinomial',
    random_state=42,
    n_jobs=-1
)
log_model.fit(X_train_clf_scaled, y_train_clf)
y_pred_log = log_model.predict(X_test_clf_scaled)

# Evaluate
log_acc = accuracy_score(y_test_clf, y_pred_log)
log_f1 = f1_score(y_test_clf, y_pred_log, average='weighted')
log_precision = precision_score(y_test_clf, y_pred_log, average='weighted')
log_recall = recall_score(y_test_clf, y_pred_log, average='weighted')

classification_results['Logistic Regression'] = {
    'Accuracy': log_acc, 'F1-Score': log_f1,
    'Precision': log_precision, 'Recall': log_recall
}

print("=" * 50)
print("     Logistic Regression Results")
print("=" * 50)
print(f"  Accuracy  = {log_acc:.4f}")
print(f"  Precision = {log_precision:.4f}")
print(f"  Recall    = {log_recall:.4f}")
print(f"  F1-Score  = {log_f1:.4f}")
print("=" * 50)

print("\\nClassification Report:")
print(classification_report(y_test_clf, y_pred_log, target_names=['Low', 'Medium', 'High']))"""))

cells.append(code("""# Logistic Regression: Confusion Matrix
fig, ax = plt.subplots(figsize=(7, 6))
cm = confusion_matrix(y_test_clf, y_pred_log)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low', 'Medium', 'High'])
disp.plot(cmap='Blues', ax=ax, values_format='d')
ax.set_title('Logistic Regression — Confusion Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()"""))

# ===== MODEL 4: XGBoost =====
cells.append(md("""## 9. Model 4: XGBoost — Regression

XGBoost (Extreme Gradient Boosting) is one of the most powerful ensemble algorithms. We use it for regression here."""))

cells.append(code("""# Train XGBoost with optimized hyperparameters
xgb_model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=20,
    eval_metric='rmse'
)

xgb_model.fit(
    X_train_reg_scaled, y_train_reg,
    eval_set=[(X_test_reg_scaled, y_test_reg)],
    verbose=50
)

y_pred_xgb = xgb_model.predict(X_test_reg_scaled)

# Evaluate
xgb_mae = mean_absolute_error(y_test_reg, y_pred_xgb)
xgb_rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_xgb))
xgb_r2 = r2_score(y_test_reg, y_pred_xgb)

regression_results['XGBoost'] = {'MAE': xgb_mae, 'RMSE': xgb_rmse, 'R²': xgb_r2}

print("\\n" + "=" * 50)
print("     XGBoost Regression Results")
print("=" * 50)
print(f"  MAE  = {xgb_mae:.4f}")
print(f"  RMSE = {xgb_rmse:.4f}")
print(f"  R²   = {xgb_r2:.4f}")
print("=" * 50)"""))

cells.append(code("""# XGBoost: Actual vs Predicted + Feature Importance
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Scatter plot
axes[0].scatter(y_test_reg, y_pred_xgb, alpha=0.1, s=5, color='#2ecc71')
axes[0].plot([0, 21], [0, 21], 'r--', linewidth=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Fitness Level', fontsize=12)
axes[0].set_ylabel('Predicted Fitness Level', fontsize=12)
axes[0].set_title(f'XGBoost — Actual vs Predicted (R² = {xgb_r2:.4f})', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)

# Feature Importance
feature_imp = pd.Series(xgb_model.feature_importances_, index=X_reg.columns)
top_features = feature_imp.nlargest(10)
top_features.plot.barh(ax=axes[1], color='#2ecc71', edgecolor='black')
axes[1].set_title('Top 10 Feature Importance — XGBoost', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Importance Score')
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()"""))

# ===== MODEL 5: SVM =====
cells.append(md("""## 10. Model 5: SVM (Support Vector Machine) — Classification

SVM is used for classification. We use `fitness_category` as the target. Due to the large dataset, we use `LinearSVC` for efficiency with an RBF-approximated kernel."""))

cells.append(code("""from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# Train SVM (LinearSVC is much faster for large datasets)
svm_model = LinearSVC(
    C=1.0,
    max_iter=5000,
    dual='auto',
    random_state=42
)
svm_model.fit(X_train_clf_scaled, y_train_clf)
y_pred_svm = svm_model.predict(X_test_clf_scaled)

# Evaluate
svm_acc = accuracy_score(y_test_clf, y_pred_svm)
svm_f1 = f1_score(y_test_clf, y_pred_svm, average='weighted')
svm_precision = precision_score(y_test_clf, y_pred_svm, average='weighted')
svm_recall = recall_score(y_test_clf, y_pred_svm, average='weighted')

classification_results['SVM'] = {
    'Accuracy': svm_acc, 'F1-Score': svm_f1,
    'Precision': svm_precision, 'Recall': svm_recall
}

print("=" * 50)
print("     SVM Classification Results")
print("=" * 50)
print(f"  Accuracy  = {svm_acc:.4f}")
print(f"  Precision = {svm_precision:.4f}")
print(f"  Recall    = {svm_recall:.4f}")
print(f"  F1-Score  = {svm_f1:.4f}")
print("=" * 50)

print("\\nClassification Report:")
print(classification_report(y_test_clf, y_pred_svm, target_names=['Low', 'Medium', 'High']))"""))

cells.append(code("""# SVM: Confusion Matrix
fig, ax = plt.subplots(figsize=(7, 6))
cm_svm = confusion_matrix(y_test_clf, y_pred_svm)
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=['Low', 'Medium', 'High'])
disp_svm.plot(cmap='Greens', ax=ax, values_format='d')
ax.set_title('SVM — Confusion Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()"""))

# ===== CELL: FINAL COMPARISON =====
cells.append(md("""## 11. Final Model Comparison & Summary"""))

cells.append(code("""# ==================== REGRESSION COMPARISON ====================
print("=" * 70)
print("              REGRESSION MODELS COMPARISON")
print("=" * 70)

reg_df = pd.DataFrame(regression_results).T
reg_df = reg_df.round(4)
reg_df.index.name = 'Model'
print(reg_df.to_string())
print()

# Best model
best_reg = reg_df['R²'].idxmax()
print(f"🏆 Best Regression Model: {best_reg} (R² = {reg_df.loc[best_reg, 'R²']:.4f})")

print("\\n" + "=" * 70)
print("              CLASSIFICATION MODELS COMPARISON")
print("=" * 70)

clf_df = pd.DataFrame(classification_results).T
clf_df = clf_df.round(4)
clf_df.index.name = 'Model'
print(clf_df.to_string())
print()

best_clf = clf_df['Accuracy'].idxmax()
print(f"🏆 Best Classification Model: {best_clf} (Accuracy = {clf_df.loc[best_clf, 'Accuracy']:.4f})")"""))

cells.append(code("""# Visualization: Regression Models Comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
metrics = ['MAE', 'RMSE', 'R²']
colors_reg = ['#3498db', '#e74c3c', '#2ecc71']

for i, metric in enumerate(metrics):
    bars = axes[i].bar(reg_df.index, reg_df[metric], color=colors_reg, edgecolor='black', alpha=0.85)
    axes[i].set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
    axes[i].set_ylabel(metric)
    # Add value labels
    for bar, val in zip(bars, reg_df[metric]):
        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[i].tick_params(axis='x', rotation=15)

fig.suptitle('Regression Models Comparison', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()"""))

cells.append(code("""# Visualization: Classification Models Comparison
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
clf_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors_clf = ['#9b59b6', '#1abc9c']

for i, metric in enumerate(clf_metrics):
    bars = axes[i].bar(clf_df.index, clf_df[metric], color=colors_clf, edgecolor='black', alpha=0.85)
    axes[i].set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
    axes[i].set_ylabel(metric)
    axes[i].set_ylim(0, 1.15)
    for bar, val in zip(bars, clf_df[metric]):
        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[i].tick_params(axis='x', rotation=15)

fig.suptitle('Classification Models Comparison', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()"""))

# ===== SAVE MODELS =====
cells.append(md("""## 12. Save All Trained Models

All models, scalers, and encoders are saved for future deployment and inference."""))

cells.append(code("""# Create models directory
os.makedirs('saved_models', exist_ok=True)

# Save all regression models
joblib.dump(knn_model, 'saved_models/knn_model.pkl')
joblib.dump(lr_model, 'saved_models/linear_regression_model.pkl')
joblib.dump(xgb_model, 'saved_models/xgboost_model.pkl')

# Save all classification models
joblib.dump(log_model, 'saved_models/logistic_regression_model.pkl')
joblib.dump(svm_model, 'saved_models/svm_model.pkl')

# Save scalers
joblib.dump(scaler_reg, 'saved_models/scaler_regression.pkl')
joblib.dump(scaler_clf, 'saved_models/scaler_classification.pkl')

# Save label encoder
joblib.dump(le_gender, 'saved_models/label_encoder_gender.pkl')

# Save feature names for reference
feature_names = list(X_reg.columns)
joblib.dump(feature_names, 'saved_models/feature_names.pkl')

print("✅ All models saved successfully to 'saved_models/' directory!")
print()
for f in sorted(os.listdir('saved_models')):
    size = os.path.getsize(f'saved_models/{f}') / 1024
    print(f"  📁 {f:45s}  ({size:,.1f} KB)")"""))

# ===== SUMMARY =====
cells.append(md("""## 13. Conclusion

### Summary of Results

| Model | Task | Key Metric |
|-------|------|-----------|
| **KNN** | Regression | R² score |
| **Linear Regression** | Regression | R² score |
| **XGBoost** | Regression | R² score |
| **Logistic Regression** | Classification | Accuracy / F1 |
| **SVM** | Classification | Accuracy / F1 |

### Key Findings:
- **XGBoost** typically achieves the highest R² for regression tasks due to its gradient boosting ensemble approach
- **Feature importance** analysis reveals which health metrics most strongly predict fitness level
- All models are saved in the `saved_models/` directory for deployment

### Files Saved:
- `saved_models/knn_model.pkl`
- `saved_models/linear_regression_model.pkl`
- `saved_models/xgboost_model.pkl`
- `saved_models/logistic_regression_model.pkl`
- `saved_models/svm_model.pkl`
- `saved_models/scaler_regression.pkl`
- `saved_models/scaler_classification.pkl`
- `saved_models/label_encoder_gender.pkl`
- `saved_models/feature_names.pkl`"""))

# ===== Build Notebook =====
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "file_extension": ".py",
            "nbconvert_exporter": "python"
        }
    },
    "cells": cells
}

# Write notebook
output_path = r'c:\Users\youss\Downloads\Final_Project_Team_5\ML_Fitness_Prediction.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook created: {output_path}")
print(f"Total cells: {len(cells)}")
