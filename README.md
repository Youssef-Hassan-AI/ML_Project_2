# 🏋️ Fitness Level Prediction — Machine Learning Project

This project leverages Machine Learning to predict an individual's fitness level based on daily health, activity, and vitals data. The application predicts both a continuous fitness score (Regression) and a categorized fitness bin (Classification) using a clean Frontend-Backend architecture.

## 🏗️ Architecture

The project is structured with a clear separation of concerns:
- **`frontend/app.py`**: A professional Gradio 6.0 interface providing interactive Dataset Explorers, EDA, Model Performance metrics, Interactive Prediction, and Batch CSV Prediction.
- **`backend/`**: Contains core logic.
  - **`model_service.py`**: Handles loading trained models and generating predictions.
  - **`preprocessing.py`**: Applies exact data imputation, one-hot encoding, label encoding, and scaling pipelines used during training.
  - **`data_service.py`**: Reads dataset statistics and samples intelligently to avoid memory overhead.
- **`saved_models/`**: Contains the pre-trained `pkl` files (XGBoost, SVM, KNN, Logistic Regression, Linear Regression) alongside feature names and scalers.

## 🧠 Models Implemented

### 1. Regression (Target: Continuous `fitness_level`)
- **XGBoost (Selected)**: R² = 0.9463
- **KNN**: R² = 0.9427
- **Linear Regression**: R² = 0.8089

### 2. Classification (Target: `fitness_category`)
- **Logistic Regression (Selected)**: Accuracy = 0.7823
- **Support Vector Machine (SVM)**: Accuracy = 0.7608

## 🚀 How to Run Locally

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Youssef-Hassan-AI/ML_Project_2.git
   cd ML_Project_2
   ```

2. **Install Dependencies:**
   Ensure you have Python 3.9+ installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   ```bash
   python frontend/app.py
   ```
   Open the provided local URL (typically `http://127.0.0.1:7861/`) in your browser.

## 📂 Project Structure

```text
├── backend/
│   ├── __init__.py
│   ├── data_service.py
│   ├── model_service.py
│   └── preprocessing.py
├── frontend/
│   └── app.py
├── saved_models/
│   ├── (Contains all .pkl models, scalers, and encoders)
├── ML_Fitness_Prediction.ipynb  # Original Jupyter Notebook pipeline
├── create_notebook.py           # Script that generated the notebook
├── eval_models.py               # Independent validation script
├── explore.py                   # Data exploration utilities
├── my_dataset.csv               # Dataset (Note: May require LFS if >100MB)
├── requirements.txt             # Python dependencies
└── README.md                    # Project Documentation
```

## 🌐 Deployment
This architecture is 100% compatible with Hugging Face Spaces using the Gradio template. Simply connect this repository to a new Hugging Face Space!
