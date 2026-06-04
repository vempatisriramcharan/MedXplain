from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import pandas as pd
import numpy as np

def evaluate_model(model, X_test, y_test):
    """Evaluates a single model and returns metrics."""
    y_pred = model.predict(X_test)
    
    # Probabilities for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred # Fallback if no predict_proba
        
    # Check if multiclass or binary based on y_test unique values
    unique_classes = len(np.unique(y_test))
    
    if unique_classes > 2:
        # Multiclass metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # ROC AUC for multiclass (requires probabilities for all classes)
        if hasattr(model, "predict_proba"):
            y_prob_multi = model.predict_proba(X_test)
            try:
                roc_auc = roc_auc_score(y_test, y_prob_multi, multi_class='ovr')
            except:
                roc_auc = np.nan
        else:
            roc_auc = np.nan
            
    else:
        # Binary metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            roc_auc = roc_auc_score(y_test, y_prob)
        except:
            roc_auc = np.nan
            
    return {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    }

def evaluate_all_models(models_dict, X_test, y_test):
    """Evaluates a dictionary of trained models and returns a DataFrame of metrics."""
    results = []
    for name, model in models_dict.items():
        metrics = evaluate_model(model, X_test, y_test)
        metrics['Model'] = name
        results.append(metrics)
        
    df_results = pd.DataFrame(results)
    # Reorder columns
    cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    df_results = df_results[cols]
    return df_results
