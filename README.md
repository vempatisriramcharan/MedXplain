# MediXplain — Explainable AI for Clinical Diagnosis

MediXplain is an end-to-end machine learning system that predicts diseases (Heart Disease, Diabetes, Liver Disorder) and provides explainable AI outputs using SHAP and LIME. The goal is not just to offer accurate medical predictions, but to build trust with clinicians through transparent model reasoning.

![MediXplain Overview](https://via.placeholder.com/800x400?text=MediXplain+Dashboard)

## 🎯 Objectives
- Multi-class and binary classification for disease prediction.
- Explainability using **SHAP** (Global & Local) and **LIME** (Local).
- Comparison of accuracy vs interpretability across multiple models.
- Interactive **Streamlit** dashboard tailored for clinicians.

## 📂 Project Structure
```text
XAI_Medical_Diagnosis/
│
├── Datasets/                 # Original medical datasets
├── app/
│   └── dashboard.py          # Streamlit clinician dashboard
├── src/
│   ├── data_preprocessing.py # Data loading, scaling, and imputation
│   ├── model_training.py     # Hyperparameter tuning and model pipelines
│   ├── evaluation.py         # Performance metrics (Accuracy, F1, ROC-AUC)
│   ├── explainability.py     # SHAP and LIME wrappers
│   └── tradeoff_analysis.py  # Accuracy vs Interpretability plotting
├── notebooks/                # Jupyter Notebooks for step-by-step execution
├── models/                   # Saved .joblib model files
├── requirements.txt          # Python dependencies
└── README.md
```

## 🚀 Setup Instructions

1. **Activate the Virtual Environment**:
   Ensure you are in the project root directory.
   ```powershell
   # Windows
   .\venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🧠 How to Run

### 1. Model Training
To train the models and generate the required `.joblib` files, you can run the provided Jupyter notebook or execute the source files interactively:
```bash
jupyter notebook notebooks/02_Model_Training.ipynb
```
*Note: Make sure to train the models before running the dashboard.*

### 2. Streamlit Dashboard
Launch the clinical dashboard using Streamlit:
```bash
streamlit run app/dashboard.py
```
Open your browser to `http://localhost:8501`. Here you can:
- Input patient vitals.
- See the prediction output.
- Analyze the model's reasoning through SHAP and LIME visual explanations.

### 3. Exploring the Notebooks
Navigate to the `notebooks/` folder to explore step-by-step executions:
- `01_EDA_and_Preprocessing.ipynb`: Understand data distributions.
- `03_SHAP_Explanations.ipynb`: See global feature importances.
- `04_LIME_Explanations.ipynb`: Explore local surrogate explanations.
- `05_Tradeoff_Analysis.ipynb`: View the accuracy vs. interpretability scatter plot.

## ⚖️ Tradeoff Analysis
Models like Logistic Regression provide high interpretability but may lack the raw accuracy of complex ensembles like XGBoost or Neural Networks. The tradeoffs are analyzed extensively in the `src/tradeoff_analysis.py` module and `notebooks/05_Tradeoff_Analysis.ipynb`.

## 🔮 Future Improvements
- Integrate deep learning model explainability (e.g., Integrated Gradients).
- Add support for Medical Image data (e.g., Chest X-rays) with CNN explainers.
- Deploy the Streamlit app to AWS or Heroku for public access.

---
**Disclaimer**: This system is meant for research and educational purposes. It should not be used as a substitute for professional medical advice or diagnosis.
