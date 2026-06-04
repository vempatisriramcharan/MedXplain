import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

def get_models():
    """Returns a dictionary of models to be trained."""
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'Neural Network': MLPClassifier(max_iter=1000, random_state=42)
    }

def get_param_grids():
    """Returns a dictionary of parameter grids for hyperparameter tuning."""
    return {
        'Logistic Regression': {
            'C': [0.1, 1, 10]
        },
        'Random Forest': {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20]
        },
        'XGBoost': {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1, 0.2]
        },
        'Neural Network': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'alpha': [0.0001, 0.001]
        }
    }

def train_and_tune_model(X_train, y_train, model_name, model, param_grid):
    """Trains and tunes a single model using GridSearchCV."""
    print(f"Training {model_name}...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    return grid_search.best_estimator_

def train_all_models(X_train, y_train, dataset_name):
    """Trains all models for a specific dataset and saves them."""
    models = get_models()
    param_grids = get_param_grids()
    trained_models = {}
    
    os.makedirs('models', exist_ok=True)
    
    for name, model in models.items():
        best_model = train_and_tune_model(X_train, y_train, name, model, param_grids[name])
        trained_models[name] = best_model
        
        # Save model
        filename = f"models/{dataset_name}_{name.replace(' ', '_').lower()}.joblib"
        joblib.dump(best_model, filename)
        
    return trained_models
