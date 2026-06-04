import shap
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np

def generate_shap_explainer(model, X_train, model_name):
    """Generates a SHAP explainer based on the model type."""
    if model_name in ['Random Forest', 'XGBoost']:
        explainer = shap.TreeExplainer(model)
    elif model_name == 'Logistic Regression':
        # LinearExplainer works better for LogisticRegression, but requires independent features
        explainer = shap.LinearExplainer(model, X_train)
    else:
        # KernelExplainer as a fallback (slow)
        # We sample the background dataset to speed it up
        background = shap.sample(X_train, 100)
        explainer = shap.KernelExplainer(model.predict, background)
    return explainer

def get_shap_values(explainer, X, model_name):
    """Calculates SHAP values."""
    if model_name in ['Random Forest', 'XGBoost', 'Logistic Regression']:
        shap_values = explainer.shap_values(X)
        # TreeExplainer for Random Forest returns a list of shap_values (one for each class)
        # We typically explain the positive class (index 1)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
    else:
        # For KernelExplainer, it returns an array
        shap_values = explainer.shap_values(X)
    return shap_values

def plot_shap_summary(shap_values, X, show=True):
    """Plots a SHAP summary plot."""
    shap.summary_plot(shap_values, X, show=show)
    if not show:
        fig = plt.gcf()
        return fig

def plot_shap_force(explainer, shap_values, X, instance_index=0, model_name=''):
    """Plots a SHAP force plot for a single instance."""
    shap.initjs() # Required for force plot in notebook
    # TreeExplainer expected value might be an array
    expected_value = explainer.expected_value
    if isinstance(expected_value, np.ndarray) or isinstance(expected_value, list):
        expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
        
    return shap.force_plot(
        expected_value, 
        shap_values[instance_index], 
        X.iloc[instance_index]
    )

def generate_lime_explainer(X_train, feature_names, class_names=['Negative', 'Positive']):
    """Generates a LIME tabular explainer."""
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )
    return explainer

def get_lime_explanation(explainer, model, X_instance, num_features=10):
    """Gets LIME explanation for a single instance."""
    predict_fn = model.predict_proba if hasattr(model, 'predict_proba') else model.predict
    exp = explainer.explain_instance(
        data_row=X_instance, 
        predict_fn=predict_fn, 
        num_features=num_features
    )
    return exp
