import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os

def load_heart_disease_data(filepath="Datasets/UCI Heart Disease (Cleveland) Dataset/heart_disease_uci.csv"):
    df = pd.read_csv(filepath)
    # Target variable 'num', values > 0 are considered heart disease (1)
    df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
    df = df.drop(columns=['id', 'dataset', 'num']) # Drop non-informative columns
    
    # Handle missing values
    categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('target')
    
    imputer_cat = SimpleImputer(strategy='most_frequent')
    imputer_num = SimpleImputer(strategy='median')
    
    df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])
    df[numerical_cols] = imputer_num.fit_transform(df[numerical_cols])
    
    # Encode categorical features
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        
    X = df.drop(columns=['target'])
    y = df['target']
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X_scaled, y, label_encoders, scaler

def load_diabetes_data(filepath="Datasets/Pima Indians Diabetes Dataset/diabetes.csv"):
    df = pd.read_csv(filepath)
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    # Replace 0s with NaNs for imputation in specific columns
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    X[cols_with_zeros] = X[cols_with_zeros].replace(0, np.nan)
    
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)
    
    return X_scaled, y, {}, scaler

def load_liver_data(filepath="Datasets/Indian Liver Patient Dataset (ILPD)/Indian Liver Patient Dataset (ILPD).csv"):
    df = pd.read_csv(filepath)
    
    # Target variable 'is_patient': 1 is patient, 2 is healthy. Convert to 1 (patient) and 0 (healthy)
    df['is_patient'] = df['is_patient'].apply(lambda x: 1 if x == 1 else 0)
    
    # Handle missing values in 'ag_ratio'
    imputer = SimpleImputer(strategy='median')
    # ag_ratio contains some missing values potentially, but we apply to all just in case
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('is_patient')
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
    
    # Encode categorical 'gender'
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])
    
    X = df.drop(columns=['is_patient'])
    y = df['is_patient']
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X_scaled, y, {'gender': le}, scaler

def load_breast_cancer_data(filepath="Datasets/Breast Cancer Wisconsin (Diagnostic) Dataset/data.csv"):
    df = pd.read_csv(filepath)
    # Target is diagnosis (M=1, B=0)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    # Drop id and any empty columns (like 'Unnamed: 32' which sometimes appears in this dataset)
    cols_to_drop = ['id']
    if 'Unnamed: 32' in df.columns:
        cols_to_drop.append('Unnamed: 32')
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Impute just in case
    imputer = SimpleImputer(strategy='median')
    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']
    
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)
    
    return X_scaled, y, {}, scaler

def load_ckd_data(filepath="Datasets/Chronic Kidney Disease (CKD) Dataset/Chronic_Kidney_Dsease_data.csv"):
    df = pd.read_csv(filepath)
    # Drop PatientID, DoctorInCharge
    df = df.drop(columns=['PatientID', 'DoctorInCharge'], errors='ignore')
    
    X = df.drop(columns=['Diagnosis'])
    y = df['Diagnosis']
    
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)
    
    return X_scaled, y, {}, scaler

def load_stroke_data(filepath="Datasets/Stroke Prediction Dataset (Kaggle)/healthcare-dataset-stroke-data.csv"):
    df = pd.read_csv(filepath)
    df = df.drop(columns=['id'], errors='ignore')
    
    # Handle 'N/A' in bmi
    df['bmi'] = df['bmi'].replace('N/A', np.nan)
    df['bmi'] = pd.to_numeric(df['bmi'])
    
    y = df['stroke']
    X = df.drop(columns=['stroke'])
    
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(exclude=['object']).columns
    
    imputer_cat = SimpleImputer(strategy='most_frequent')
    imputer_num = SimpleImputer(strategy='median')
    
    X[categorical_cols] = imputer_cat.fit_transform(X[categorical_cols])
    X[numerical_cols] = imputer_num.fit_transform(X[numerical_cols])
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X_scaled, y, label_encoders, scaler
