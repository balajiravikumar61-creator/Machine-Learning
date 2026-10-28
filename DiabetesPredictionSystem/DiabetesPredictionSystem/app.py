"""
Diabetes Prediction System Using Machine Learning
==================================================
Script: app.py
Framework: Flask
Description: Local web application serving:
  - Fig. 1: Dataset Overview & Schema
  - Fig. 2: Data Preprocessing & Invalid Zero Imputation
  - Fig. 3: Exploratory Data Analysis (EDA) Interactive Gallery
  - Fig. 4: Multi-Model Training & Benchmark Comparison
  - Fig. 5: Held-Out Model Evaluation Metrics & ROC / Confusion Matrix
  - Fig. 6: Interactive Prediction Form with Live Pipeline Inference

Educational Disclaimer:
  This application is developed strictly for educational demonstration and machine
  learning coursework. It must not be presented or utilized as a medical diagnostic tool.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, flash, redirect, url_for

# Initialize Flask application
app = Flask(__name__)
app.secret_key = "diabetes-ml-educational-project-secret-key"

# Directory references
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'static', 'plots')
MODEL_FILE = os.path.join(MODELS_DIR, 'diabetes_model.pkl')
METRICS_FILE = os.path.join(MODELS_DIR, 'model_metrics.json')
DATASET_INFO_FILE = os.path.join(MODELS_DIR, 'dataset_info.json')

# Expected feature column names matching trained pipeline
FEATURE_NAMES = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age'
]


def check_and_initialize_pipeline():
    """Ensure trained model, metadata, and EDA plots exist; run pipeline if not."""
    required_files = [
        MODEL_FILE,
        METRICS_FILE,
        DATASET_INFO_FILE,
        os.path.join(PLOTS_DIR, 'outcome_distribution.png'),
        os.path.join(PLOTS_DIR, 'glucose_vs_outcome.png'),
        os.path.join(PLOTS_DIR, 'bmi_distribution.png'),
        os.path.join(PLOTS_DIR, 'age_distribution.png'),
        os.path.join(PLOTS_DIR, 'correlation_heatmap.png'),
        os.path.join(PLOTS_DIR, 'confusion_matrix.png'),
        os.path.join(PLOTS_DIR, 'roc_curve.png')
    ]
    
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print("[App Setup] Required model files or plots not detected. Initializing training pipeline...")
        import train_model
        train_model.run_pipeline()


def load_project_metadata():
    """Load cached dataset info and benchmark metrics JSON."""
    check_and_initialize_pipeline()
    
    with open(METRICS_FILE, 'r', encoding='utf-8') as f:
        metrics_data = json.load(f)
        
    with open(DATASET_INFO_FILE, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)
        
    return metrics_data, dataset_info


@app.route('/', methods=['GET'])
def index():
    """Render the main project dashboard with all 6 required sections."""
    metrics_data, dataset_info = load_project_metadata()
    
    return render_template(
        'index.html',
        metrics=metrics_data,
        dataset=dataset_info,
        prediction_result=None,
        form_data={}
    )


@app.route('/predict', methods=['POST'])
def predict():
    """Process prediction form inputs, validate, execute saved pipeline, and display results."""
    metrics_data, dataset_info = load_project_metadata()
    form_data = request.form.to_dict()
    
    # Input validation
    errors = []
    parsed_values = {}

    validation_rules = {
        'Pregnancies': {'type': int, 'min': 0, 'max': 25, 'label': 'Pregnancies'},
        'Glucose': {'type': float, 'min': 1.0, 'max': 500.0, 'label': 'Glucose Level'},
        'BloodPressure': {'type': float, 'min': 1.0, 'max': 300.0, 'label': 'Blood Pressure'},
        'SkinThickness': {'type': float, 'min': 0.0, 'max': 120.0, 'label': 'Skin Thickness'},
        'Insulin': {'type': float, 'min': 0.0, 'max': 1000.0, 'label': 'Insulin Level'},
        'BMI': {'type': float, 'min': 5.0, 'max': 90.0, 'label': 'Body Mass Index (BMI)'},
        'DiabetesPedigreeFunction': {'type': float, 'min': 0.01, 'max': 3.5, 'label': 'Diabetes Pedigree Function'},
        'Age': {'type': int, 'min': 1, 'max': 130, 'label': 'Age'}
    }

    for field, rule in validation_rules.items():
        raw_val = form_data.get(field, '').strip()
        if not raw_val:
            errors.append(f"Please provide a valid value for {rule['label']}.")
            continue
        try:
            val = rule['type'](raw_val)
            if val < rule['min'] or val > rule['max']:
                errors.append(f"{rule['label']} must be between {rule['min']} and {rule['max']}.")
            else:
                parsed_values[field] = val
        except (ValueError, TypeError):
            errors.append(f"Invalid numeric input entered for {rule['label']}.")

    if errors:
        for err in errors:
            flash(err, 'danger')
        return render_template(
            'index.html',
            metrics=metrics_data,
            dataset=dataset_info,
            prediction_result=None,
            form_data=form_data
        ), 400

    try:
        # Load the saved trained pipeline (includes SimpleImputer, StandardScaler, and Classifier)
        pipeline = joblib.load(MODEL_FILE)

        # Prepare single record DataFrame with matching feature names
        input_df = pd.DataFrame([parsed_values], columns=FEATURE_NAMES)

        # Note: In the Pima dataset, measurement zero values (e.g. SkinThickness=0, Insulin=0)
        # represent unmeasured readings. We convert 0 in these fields to NaN so the pipeline's
        # median imputer safely handles them.
        for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
            if input_df[col].iloc[0] == 0:
                input_df[col] = np.nan

        # Generate model prediction and probability
        pred_class = int(pipeline.predict(input_df)[0])
        
        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(input_df)[0]
            positive_prob = float(probabilities[1]) * 100.0
            negative_prob = float(probabilities[0]) * 100.0
            display_prob = positive_prob if pred_class == 1 else negative_prob
            prob_str = f"{display_prob:.2f}%"
            pos_prob_str = f"{positive_prob:.2f}%"
        else:
            prob_str = "N/A"
            pos_prob_str = "N/A"

        class_label = "Diabetes-positive class" if pred_class == 1 else "Diabetes-negative class"

        prediction_result = {
            'predicted_class': pred_class,
            'prediction_label': class_label,
            'estimated_probability': prob_str,
            'positive_probability': pos_prob_str,
            'selected_model': metrics_data.get('selected_model', 'Trained Classifier'),
            'inputs': parsed_values
        }

        flash(f"Prediction generated successfully using the {metrics_data.get('selected_model')} pipeline.", 'success')

    except Exception as e:
        flash(f"Error during model inference: {str(e)}", 'danger')
        prediction_result = None

    return render_template(
        'index.html',
        metrics=metrics_data,
        dataset=dataset_info,
        prediction_result=prediction_result,
        form_data=form_data
    )


if __name__ == '__main__':
    # Local execution only; no cloud dependencies or external APIs
    check_and_initialize_pipeline()
    print("\n" + "=" * 65)
    print("Diabetes Prediction System Local Web Server Running")
    print("Access application at: http://127.0.0.1:5000")
    print("=" * 65 + "\n")
    app.run(host='127.0.0.1', port=5000, debug=True)
