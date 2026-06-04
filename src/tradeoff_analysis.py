import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_interpretability_scores():
    """Returns a dictionary of heuristic interpretability scores for models (0-100 scale)."""
    return {
        'Logistic Regression': 90, # Highly interpretable (coefficients)
        'Random Forest': 60,       # Moderately interpretable (feature importance, trees)
        'XGBoost': 50,             # Less interpretable than RF (more complex ensembles)
        'Neural Network': 20       # Black box
    }

def analyze_tradeoff(df_results):
    """Analyzes tradeoff between accuracy and interpretability."""
    # Ensure Model column is set as index or exists
    interpretability = get_interpretability_scores()
    
    # Map interpretability scores to the results dataframe
    df_tradeoff = df_results.copy()
    df_tradeoff['Interpretability'] = df_tradeoff['Model'].map(interpretability)
    
    return df_tradeoff

def plot_tradeoff(df_tradeoff):
    """Plots Accuracy vs Interpretability."""
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    sns.scatterplot(
        data=df_tradeoff, 
        x='Interpretability', 
        y='Accuracy', 
        hue='Model', 
        s=200, 
        palette='viridis'
    )
    
    # Annotate points
    for i in range(df_tradeoff.shape[0]):
        plt.text(
            x=df_tradeoff['Interpretability'][i] + 1, 
            y=df_tradeoff['Accuracy'][i], 
            s=df_tradeoff['Model'][i], 
            fontdict=dict(size=10)
        )
        
    plt.title('Model Tradeoff: Accuracy vs Interpretability', fontsize=16)
    plt.xlabel('Interpretability Score (Heuristic 0-100)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    fig = plt.gcf()
    return fig
