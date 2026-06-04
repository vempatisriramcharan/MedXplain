import sys
sys.path.append('.')
from src.data_preprocessing import (
    load_heart_disease_data, 
    load_diabetes_data, 
    load_liver_data,
    load_breast_cancer_data,
    load_ckd_data,
    load_stroke_data
)
from src.model_training import train_all_models

def main():
    datasets = {
        "Heart Disease": load_heart_disease_data,
        "Diabetes": load_diabetes_data,
        "Liver Disorder": load_liver_data,
        "Breast Cancer": load_breast_cancer_data,
        "Chronic Kidney Disease": load_ckd_data,
        "Stroke Prediction": load_stroke_data
    }
    
    for name, loader in datasets.items():
        print(f"\\n--- Training models for {name} ---")
        try:
            # Loader returns varying number of items, we only need X and y
            result = loader()
            X, y = result[0], result[1]
            train_all_models(X, y, name)
            print(f"Successfully trained models for {name}.")
        except Exception as e:
            print(f"Failed to train models for {name}: {e}")

if __name__ == '__main__':
    main()
