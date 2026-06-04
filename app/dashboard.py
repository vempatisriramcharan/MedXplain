import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os
from PIL import Image

# Add src to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_preprocessing import (
    load_heart_disease_data, 
    load_diabetes_data, 
    load_liver_data,
    load_breast_cancer_data,
    load_ckd_data,
    load_stroke_data
)
from src.explainability import generate_shap_explainer, get_shap_values, plot_shap_summary, generate_lime_explainer, get_lime_explanation

# Configuration
st.set_page_config(page_title="MediXplain | Clinical Diagnosis", page_icon="🏥", layout="wide")

# Custom CSS for Premium, Clinician-friendly Aesthetic
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        color: #2c3e50;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.5);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(118, 75, 162, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(118, 75, 162, 0.4);
    }
    
    /* Headers */
    h1 {
        background: -webkit-linear-gradient(45deg, #2c3e50, #3498db);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
    }
    
    h2, h3 {
        color: #34495e;
        font-weight: 600 !important;
    }
    
    /* Input Fields */
    .stNumberInput>div>div>input, .stSelectbox>div>div>div {
        border-radius: 8px;
        border: 1px solid rgba(0, 0, 0, 0.1);
        background: rgba(255, 255, 255, 0.9);
        transition: all 0.2s;
    }
    .stNumberInput>div>div>input:focus, .stSelectbox>div>div>div:focus {
        border-color: #764ba2;
        box-shadow: 0 0 0 2px rgba(118, 75, 162, 0.2);
    }
    
    /* Prediction Box (Glassmorphism & Pulse) */
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(220, 53, 69, 0); }
        100% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0); }
    }
    @keyframes pulse-green {
        0% { box-shadow: 0 0 0 0 rgba(25, 135, 84, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(25, 135, 84, 0); }
        100% { box-shadow: 0 0 0 0 rgba(25, 135, 84, 0); }
    }
    
    .prediction-box {
        padding: 30px;
        border-radius: 16px;
        margin: 30px 0;
        text-align: center;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: all 0.3s ease;
    }
    
    .positive-pred {
        background: rgba(255, 235, 238, 0.85);
        color: #c62828;
        border: 1px solid rgba(229, 115, 115, 0.5);
        animation: pulse-red 2s infinite;
    }
    
    .negative-pred {
        background: rgba(232, 245, 233, 0.85);
        color: #2e7d32;
        border: 1px solid rgba(129, 199, 132, 0.5);
        animation: pulse-green 2s infinite;
    }
    
    .prediction-box h2 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700 !important;
        background: transparent !important;
        -webkit-text-fill-color: inherit !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.5);
        border-bottom: 3px solid #764ba2;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to get model path
def get_model_path(dataset_name, model_name):
    return f"models/{dataset_name}_{model_name.replace(' ', '_').lower()}.joblib"

# Sidebar
st.sidebar.title("🏥 MediXplain Settings")
dataset_choices = [
    "Heart Disease", 
    "Diabetes", 
    "Liver Disorder", 
    "Breast Cancer", 
    "Chronic Kidney Disease", 
    "Stroke Prediction"
]
dataset_choice = st.sidebar.selectbox("Select Disease Dataset", dataset_choices)
model_choice = st.sidebar.selectbox("Select Prediction Model", ["Logistic Regression", "Random Forest", "XGBoost", "Neural Network"])

# Main Header
st.title("MediXplain — Explainable AI for Clinical Diagnosis")
st.markdown("Enter patient vitals below to receive a prediction and understand the AI's reasoning.")

# Load Data based on choice to get feature names and scalers
@st.cache_data
def load_and_prep_data(choice):
    if choice == "Heart Disease":
        X, y, le, scaler = load_heart_disease_data()
        return X, y, le, scaler
    elif choice == "Diabetes":
        X, y, le, scaler = load_diabetes_data()
        return X, y, le, scaler
    elif choice == "Liver Disorder":
        X, y, le, scaler = load_liver_data()
        return X, y, le, scaler
    elif choice == "Breast Cancer":
        X, y, le, scaler = load_breast_cancer_data()
        return X, y, le, scaler
    elif choice == "Chronic Kidney Disease":
        X, y, le, scaler = load_ckd_data()
        return X, y, le, scaler
    elif choice == "Stroke Prediction":
        X, y, le, scaler = load_stroke_data()
        return X, y, le, scaler

X_train, y_train, label_encoders, scaler = load_and_prep_data(dataset_choice)

# Load Model
model_path = get_model_path(dataset_choice, model_choice)
try:
    model = joblib.load(model_path)
    model_loaded = True
except FileNotFoundError:
    st.error(f"Model file not found: {model_path}. Please train the models first by running notebooks/02_Model_Training.ipynb.")
    model_loaded = False

if model_loaded:
    st.subheader("Patient Vitals Input")
    col1, col2, col3 = st.columns(3)
    
    input_data = {}
    features = X_train.columns
    
    # Generate dynamic input fields
    for i, feature in enumerate(features):
        with [col1, col2, col3][i % 3]:
            if label_encoders and feature in label_encoders:
                le = label_encoders[feature]
                # Filter out any non-string representations if needed, but classes_ should be fine
                options = le.classes_
                selected = st.selectbox(f"{feature}", options=options)
                val = le.transform([selected])[0]
                input_data[feature] = val
            else:
                val = st.number_input(f"{feature}", value=0.0)
                input_data[feature] = val

    if st.button("Predict & Explain"):
        # Create DataFrame from input
        input_df = pd.DataFrame([input_data])
        
        # Scale the input using the loaded scaler
        # We need the original columns order
        input_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
        
        with st.spinner("Analyzing patient data..."):
            # Prediction
            prediction = model.predict(input_scaled)[0]
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_scaled)[0][1] # Probability of positive class
            else:
                prob = None
                
            # Display Prediction
            pred_text = "Positive for " + dataset_choice if prediction == 1 else "Negative for " + dataset_choice
            pred_class = "positive-pred" if prediction == 1 else "negative-pred"
            
            prob_text = f" (Confidence: {prob*100:.1f}%)" if prob is not None else ""
            
            st.markdown(f"""
                <div class="prediction-box {pred_class}">
                    <h2>Prediction: {pred_text}{prob_text}</h2>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("Brain of the AI (Explainability)")
            
            tab1, tab2 = st.tabs(["LIME (Local Explanation)", "SHAP (Global/Local)"])
            
            with tab1:
                st.markdown("**LIME** explains this specific patient's prediction by perturbed sampling around this instance.")
                lime_explainer = generate_lime_explainer(X_train, features)
                try:
                    lime_exp = get_lime_explanation(lime_explainer, model, input_scaled.iloc[0])
                    # Streamlit can render HTML
                    st.components.v1.html(lime_exp.as_html(), height=400, scrolling=True)
                except Exception as e:
                    st.warning(f"LIME explanation failed for this model type. {str(e)}")
            
            with tab2:
                st.markdown("**SHAP** calculates the contribution of each feature to the prediction.")
                try:
                    shap_explainer = generate_shap_explainer(model, X_train, model_choice)
                    shap_values = get_shap_values(shap_explainer, input_scaled, model_choice)
                    
                    # For Force Plot we might need JS, which is tricky in Streamlit natively without html wrapper
                    st.info("SHAP values calculated successfully. A visual representation can be complex in Streamlit natively, but you can view feature importance in the notebooks.")
                    
                    # Let's show a simple bar chart of feature contributions for this local instance
                    if isinstance(shap_values, list):
                        instance_shap = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
                    else:
                        instance_shap = shap_values[0]
                        
                    shap_df = pd.DataFrame({
                        'Feature': features,
                        'SHAP Value': instance_shap
                    }).sort_values('SHAP Value', key=abs, ascending=False)
                    
                    st.bar_chart(shap_df.set_index('Feature'))
                except Exception as e:
                    st.warning(f"SHAP explanation encountered an error: {str(e)}")

st.markdown("---")
st.markdown("<small>Disclaimer: This tool is for demonstration and research purposes only. It should not replace professional medical advice.</small>", unsafe_allow_html=True)
