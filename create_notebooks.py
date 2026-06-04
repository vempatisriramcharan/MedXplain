import nbformat as nbf
import os

def create_eda_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# 01 - EDA and Preprocessing\nThis notebook demonstrates how to load, clean, and preprocess the datasets."),
        nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport sys\nsys.path.append('../')\nfrom src.data_preprocessing import load_heart_disease_data, load_diabetes_data, load_liver_data"),
        nbf.v4.new_markdown_cell("## Heart Disease Dataset"),
        nbf.v4.new_code_cell("X_heart, y_heart, le_heart, scaler_heart = load_heart_disease_data('../Datasets/UCI Heart Disease (Cleveland) Dataset/heart_disease_uci.csv')\ndisplay(X_heart.head())\ndisplay(y_heart.value_counts())")
    ]
    with open('notebooks/01_EDA_and_Preprocessing.ipynb', 'w') as f:
        nbf.write(nb, f)

def create_model_training_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# 02 - Model Training\nThis notebook trains various models and saves them."),
        nbf.v4.new_code_cell("import sys\nsys.path.append('../')\nfrom src.data_preprocessing import load_heart_disease_data\nfrom src.model_training import train_all_models\nfrom src.evaluation import evaluate_all_models"),
        nbf.v4.new_code_cell("X_heart, y_heart, le_heart, scaler_heart = load_heart_disease_data('../Datasets/UCI Heart Disease (Cleveland) Dataset/heart_disease_uci.csv')"),
        nbf.v4.new_code_cell("models = train_all_models(X_heart, y_heart, 'Heart Disease')\nprint('Models Trained Successfully!')"),
        nbf.v4.new_code_cell("results = evaluate_all_models(models, X_heart, y_heart)\ndisplay(results)")
    ]
    with open('notebooks/02_Model_Training.ipynb', 'w') as f:
        nbf.write(nb, f)

def create_shap_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# 03 - SHAP Explanations\nVisualizing global and local feature importance using SHAP."),
        nbf.v4.new_code_cell("import sys\nimport joblib\nsys.path.append('../')\nfrom src.data_preprocessing import load_heart_disease_data\nfrom src.explainability import generate_shap_explainer, get_shap_values, plot_shap_summary, plot_shap_force\nimport shap"),
        nbf.v4.new_code_cell("X_heart, y_heart, le_heart, scaler_heart = load_heart_disease_data('../Datasets/UCI Heart Disease (Cleveland) Dataset/heart_disease_uci.csv')\nmodel = joblib.load('../models/Heart Disease_random_forest.joblib')"),
        nbf.v4.new_code_cell("explainer = generate_shap_explainer(model, X_heart, 'Random Forest')\nshap_values = get_shap_values(explainer, X_heart, 'Random Forest')"),
        nbf.v4.new_code_cell("plot_shap_summary(shap_values, X_heart)")
    ]
    with open('notebooks/03_SHAP_Explanations.ipynb', 'w') as f:
        nbf.write(nb, f)

def create_lime_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# 04 - LIME Explanations\nExplaining individual instances using LIME."),
        nbf.v4.new_code_cell("import sys\nimport joblib\nsys.path.append('../')\nfrom src.data_preprocessing import load_heart_disease_data\nfrom src.explainability import generate_lime_explainer, get_lime_explanation"),
        nbf.v4.new_code_cell("X_heart, y_heart, le_heart, scaler_heart = load_heart_disease_data('../Datasets/UCI Heart Disease (Cleveland) Dataset/heart_disease_uci.csv')\nmodel = joblib.load('../models/Heart Disease_random_forest.joblib')"),
        nbf.v4.new_code_cell("explainer = generate_lime_explainer(X_heart, list(X_heart.columns))\nexp = get_lime_explanation(explainer, model, X_heart.iloc[0])\nexp.show_in_notebook(show_table=True)")
    ]
    with open('notebooks/04_LIME_Explanations.ipynb', 'w') as f:
        nbf.write(nb, f)

def create_tradeoff_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# 05 - Accuracy vs Interpretability Tradeoff\nComparing model performance against their interpretability."),
        nbf.v4.new_code_cell("import sys\nimport joblib\nsys.path.append('../')\nfrom src.data_preprocessing import load_heart_disease_data\nfrom src.evaluation import evaluate_all_models\nfrom src.tradeoff_analysis import analyze_tradeoff, plot_tradeoff"),
        nbf.v4.new_code_cell("X_heart, y_heart, le_heart, scaler_heart = load_heart_disease_data('../Datasets/UCI Heart Disease (Cleveland) Dataset/heart_disease_uci.csv')\n\nmodels = {\n    'Logistic Regression': joblib.load('../models/Heart Disease_logistic_regression.joblib'),\n    'Random Forest': joblib.load('../models/Heart Disease_random_forest.joblib'),\n    'XGBoost': joblib.load('../models/Heart Disease_xgboost.joblib'),\n    'Neural Network': joblib.load('../models/Heart Disease_neural_network.joblib')\n}\n\nresults = evaluate_all_models(models, X_heart, y_heart)"),
        nbf.v4.new_code_cell("tradeoff_df = analyze_tradeoff(results)\nplot_tradeoff(tradeoff_df)")
    ]
    with open('notebooks/05_Tradeoff_Analysis.ipynb', 'w') as f:
        nbf.write(nb, f)

if __name__ == '__main__':
    os.makedirs('notebooks', exist_ok=True)
    create_eda_notebook()
    create_model_training_notebook()
    create_shap_notebook()
    create_lime_notebook()
    create_tradeoff_notebook()
    print("All notebooks created successfully!")
