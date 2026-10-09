import gradio as gr
import pandas as pd
import sys
import os

# Ensure backend module is reachable
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from backend.data_service import get_dataset_stats, get_dataset_sample, get_categorical_distributions
from backend.model_service import get_model_results, predict_single, predict_batch

# Define custom theme
custom_theme = gr.themes.Default(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.sky,
    neutral_hue=gr.themes.colors.slate,
).set(
    body_background_fill="#FFFFFF",
    body_text_color="#1F2937",
    color_accent_soft="#95BAFE",
    button_primary_background_fill="#1B75BB",
    button_primary_background_fill_hover="#155D96",
    button_primary_text_color="#FFFFFF",
    button_secondary_background_fill="#CCFED8",
    button_secondary_background_fill_hover="#A0F0B3",
    button_secondary_text_color="#1F2937",
    border_color_primary="#E5E7EB",
    block_background_fill="#F9FAFB",
    block_border_width="1px",
    block_radius="8px",
    slider_color="#1B75BB"
)

css = """
.gradio-container {
    font-family: 'Inter', 'Roboto', sans-serif !important;
}
.header-title {
    color: #1B75BB !important;
    font-weight: 800;
    text-align: center;
    margin-top: 10px;
    margin-bottom: 20px;
    font-size: 2.5em;
}
.sub-title {
    color: #1F2937 !important;
    text-align: center;
    font-size: 1.2em;
    margin-bottom: 30px;
}
.stat-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}
.stat-value {
    font-size: 2em;
    font-weight: bold;
    color: #1B75BB;
}
"""

def create_home_tab():
    with gr.Tab("🏠 Home"):
        gr.Markdown("<h1 class='header-title'>Fitness Level Prediction Analytics</h1>")
        gr.Markdown("<p class='sub-title'>Professional Machine Learning Pipeline for Health & Fitness Data</p>")
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Project Objective")
                gr.Markdown(
                    "This application leverages Machine Learning to predict an individual's fitness level based on daily health, activity, and vitals data. "
                    "The system provides both continuous Regression predictions (exact fitness score) and Categorical Classification (Low, Medium, High). "
                    "\n\n**Prediction Targets:**\n"
                    "- **Continuous Fitness Level**: Raw score indicating fitness (0 - 20.63)\n"
                    "- **Fitness Category**: Binned classes [Low, Medium, High]\n"
                )
            with gr.Column(scale=1):
                stats = get_dataset_stats()
                if stats:
                    gr.HTML(f'''
                    <div class="stat-card">
                        <div class="stat-value">{stats["total_rows_approx"]:,}+</div>
                        <div>Records Analyzed</div>
                    </div>
                    <div class="stat-card" style="margin-top: 15px;">
                        <div class="stat-value">{stats["columns"]}</div>
                        <div>Features</div>
                    </div>
                    ''')
        
        gr.Markdown("### 🧠 Implemented Models")
        gr.Markdown(
            "- **Regression:** XGBoost, Linear Regression, K-Nearest Neighbors (KNN)\n"
            "- **Classification:** Logistic Regression, Support Vector Machine (SVM)\n"
        )
        
def create_dataset_explorer_tab():
    with gr.Tab("🔍 Dataset Explorer"):
        gr.Markdown("### Dataset Overview")
        stats = get_dataset_stats()
        
        if not stats:
            gr.Warning("Dataset not found. Please ensure the dataset file is present.")
            return

        with gr.Row():
            gr.Markdown(f"**Total Records:** ~{stats['total_rows_approx']:,}")
            gr.Markdown(f"**Columns:** {stats['columns']}")
            gr.Markdown(f"**Missing Values:** {stats['missing_values']} (in sample)")
            
        with gr.Accordion("View Raw Dataset Sample (1000 records)", open=False):
            df_sample = get_dataset_sample(1000)
            gr.Dataframe(value=df_sample, interactive=False)
            
        with gr.Accordion("Summary Statistics (Numerical)", open=False):
            gr.Dataframe(value=stats['summary'], interactive=False)
            
        gr.Markdown("### Categorical Distributions (Sample)")
        dists = get_categorical_distributions()
        cols = gr.Row()
        with cols:
            for k, df_dist in dists.items():
                with gr.Column():
                    gr.Markdown(f"**{k}**")
                    gr.Dataframe(value=df_dist, interactive=False)

def create_model_performance_tab():
    with gr.Tab("📈 Model Performance"):
        results = get_model_results()
        
        gr.Markdown("### 1. Regression Models (Target: Continuous Fitness Level)")
        reg_df = pd.DataFrame(results['regression'])
        
        def highlight_best(row):
            if row['Selected']:
                return ['background-color: #CCFED8'] * len(row)
            return [''] * len(row)
            
        # Try styling if Gradio allows, else just show dataframe
        gr.Dataframe(value=reg_df, interactive=False)
        gr.Markdown("🏆 **Selected Model:** XGBoost (Highest R² score due to gradient boosting ensemble power)")
        
        gr.Markdown("### 2. Classification Models (Target: Fitness Category)")
        clf_df = pd.DataFrame(results['classification'])
        gr.Dataframe(value=clf_df, interactive=False)
        gr.Markdown("🏆 **Selected Model:** Logistic Regression (Best balance of accuracy and efficiency)")
        
def create_interactive_prediction_tab():
    with gr.Tab("⚡ Interactive Prediction"):
        gr.Markdown("### Make a Prediction")
        
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    target_type = gr.Radio(choices=["Regression (Exact Score)", "Classification (Category)"], value="Regression (Exact Score)", label="Prediction Task")
                    model_sel = gr.Dropdown(choices=["XGBoost", "Linear Regression", "KNN"], value="XGBoost", label="Model")
                
                # Input features
                with gr.Row():
                    age = gr.Slider(15, 90, value=30, label="Age", step=1)
                    gender = gr.Dropdown(choices=["Male", "Female"], value="Male", label="Gender")
                    height = gr.Number(value=175, label="Height (cm)")
                    weight = gr.Number(value=70, label="Weight (kg)")
                    
                with gr.Row():
                    duration = gr.Number(value=45, label="Duration (minutes)")
                    intensity = gr.Dropdown(choices=["Low", "Medium", "High"], value="Medium", label="Intensity")
                    calories = gr.Number(value=300, label="Calories Burned")
                    steps = gr.Slider(0, 30000, value=8000, label="Daily Steps", step=100)
                    
                with gr.Row():
                    hr = gr.Number(value=110, label="Avg Heart Rate")
                    rest_hr = gr.Number(value=70, label="Resting Heart Rate")
                    bp_sys = gr.Number(value=120, label="Blood Pressure Systolic")
                    bp_dia = gr.Number(value=80, label="Blood Pressure Diastolic")
                    
                with gr.Row():
                    endurance = gr.Slider(0, 10, value=5, label="Endurance Level", step=0.1)
                    sleep = gr.Slider(0, 14, value=7, label="Sleep Hours", step=0.5)
                    stress = gr.Slider(1, 10, value=5, label="Stress Level", step=1)
                    hydration = gr.Slider(0, 5, value=2, label="Hydration (Liters)", step=0.1)
                    
                with gr.Row():
                    activity = gr.Dropdown(choices=["Cycling", "Dancing", "HIIT", "Running", "Swimming", "Tennis", "Walking", "Weight Training", "Yoga"], value="Walking", label="Activity Type")
                    smoking = gr.Dropdown(choices=["Current", "Former", "Never"], value="Never", label="Smoking Status")
                    health = gr.Dropdown(choices=["None", "Diabetes", "Hypertension"], value="None", label="Health Condition")

                predict_btn = gr.Button("Predict", variant="primary")
                
            with gr.Column(scale=1):
                gr.Markdown("### Prediction Result")
                output_res = gr.Textbox(label="Result", text_align="center", scale=2)
                
        def update_models(task):
            if "Regression" in task:
                return gr.Dropdown(choices=["XGBoost", "Linear Regression", "KNN"], value="XGBoost")
            else:
                return gr.Dropdown(choices=["Logistic Regression", "SVM"], value="Logistic Regression")
                
        target_type.change(fn=update_models, inputs=[target_type], outputs=[model_sel])
        
        def do_predict(ttype, mod, age, gen, h, w, dur, intst, cal, steps, hr, rhr, bps, bpd, endu, slp, strs, hyd, act, smk, hlth):
            # Derive bmi internally
            bmi = w / ((h/100)**2) if h > 0 else 0
            
            input_dict = {
                'age': age, 'gender': gen, 'height_cm': h, 'weight_kg': w, 'bmi': bmi,
                'duration_minutes': dur, 'intensity': intst, 'calories_burned': cal,
                'daily_steps': steps, 'avg_heart_rate': hr, 'resting_heart_rate': rhr,
                'blood_pressure_systolic': bps, 'blood_pressure_diastolic': bpd,
                'endurance_level': endu, 'sleep_hours': slp, 'stress_level': strs,
                'hydration_level': hyd, 'activity_type': act, 'smoking_status': smk,
                'health_condition': hlth
            }
            
            t = 'regression' if 'Regression' in ttype else 'classification'
            try:
                res = predict_single(input_dict, mod, t)
                if t == 'regression':
                    return f"{res:.2f} (Continuous Fitness Score)"
                else:
                    return f"Category: {res}"
            except Exception as e:
                return f"Error: {str(e)}"
                
        predict_btn.click(
            fn=do_predict,
            inputs=[target_type, model_sel, age, gender, height, weight, duration, intensity, calories, steps, hr, rest_hr, bp_sys, bp_dia, endurance, sleep, stress, hydration, activity, smoking, health],
            outputs=[output_res]
        )

def create_batch_prediction_tab():
    with gr.Tab("📁 Batch Prediction"):
        gr.Markdown("### Batch Predict via CSV")
        gr.Markdown("Upload a CSV file with the same features as the training data to get predictions for all rows.")
        
        with gr.Row():
            with gr.Column():
                target_type_batch = gr.Radio(choices=["Regression (Exact Score)", "Classification (Category)"], value="Regression (Exact Score)", label="Prediction Task")
                model_sel_batch = gr.Dropdown(choices=["XGBoost", "Linear Regression", "KNN"], value="XGBoost", label="Model")
                
                file_in = gr.File(label="Upload CSV", file_types=[".csv"])
                batch_btn = gr.Button("Run Batch Prediction", variant="primary")
            
            with gr.Column():
                out_file = gr.File(label="Download Predictions", interactive=False)
                out_msg = gr.Textbox(label="Status")
                
        def update_models_batch(task):
            if "Regression" in task:
                return gr.Dropdown(choices=["XGBoost", "Linear Regression", "KNN"], value="XGBoost")
            else:
                return gr.Dropdown(choices=["Logistic Regression", "SVM"], value="Logistic Regression")
                
        target_type_batch.change(fn=update_models_batch, inputs=[target_type_batch], outputs=[model_sel_batch])
        
        def run_batch(ttype, mod, file_obj):
            if not file_obj:
                return None, "Please upload a file."
            try:
                df = pd.read_csv(file_obj.name)
                # Recalculate BMI if missing
                if 'bmi' not in df.columns and 'weight_kg' in df.columns and 'height_cm' in df.columns:
                    df['bmi'] = df['weight_kg'] / ((df['height_cm']/100)**2)
                    
                t = 'regression' if 'Regression' in ttype else 'classification'
                preds = predict_batch(df, mod, t)
                
                df['Prediction'] = preds
                out_path = "batch_predictions.csv"
                df.to_csv(out_path, index=False)
                return out_path, f"Successfully predicted {len(df)} rows."
                
            except Exception as e:
                return None, f"Error: {str(e)}"
                
        batch_btn.click(fn=run_batch, inputs=[target_type_batch, model_sel_batch, file_in], outputs=[out_file, out_msg])

def main():
    with gr.Blocks(title="Fitness ML Pipeline") as app:
        create_home_tab()
        create_dataset_explorer_tab()
        create_model_performance_tab()
        create_interactive_prediction_tab()
        create_batch_prediction_tab()
        
    app.launch(server_name="127.0.0.1", server_port=7861, show_error=True, theme=custom_theme, css=css)

if __name__ == "__main__":
    main()
