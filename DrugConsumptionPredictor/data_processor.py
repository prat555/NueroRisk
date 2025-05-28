import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import streamlit as st
import os
from io import StringIO

# Define the path to the dataset
DEFAULT_DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00373/drug_consumption.data"

def load_data():
    """
    Load the drug consumption dataset from UCI repository.
    Returns a pandas DataFrame with the dataset.
    """
    try:
        # Load data from URL
        data = pd.read_csv(DEFAULT_DATASET_URL, header=None)
        
        # Define column names based on dataset documentation
        column_names = [
            'ID', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity',
            'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 
            'SS', 'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 
            'Choc', 'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 
            'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA'
        ]
        
        data.columns = column_names
        
        # Process the dataset
        return preprocess_data(data)
    except Exception as e:
        st.error(f"Error loading default dataset: {e}")
        
        # If default loading fails, offer user upload option
        st.info("Default dataset could not be loaded. Please upload a dataset file.")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                return preprocess_data(data)
            except Exception as e:
                st.error(f"Error processing uploaded file: {e}")
                return None
        return None

def preprocess_data(data):
    """
    Preprocess the drug consumption dataset:
    - Convert categorical variables
    - Handle missing values
    - Standardize numerical features
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Map drug consumption classes to binary (CL0 and CL1 = "Never Used", CL2-CL6 = "Used")
    drug_columns = ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 
                    'Choc', 'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 
                    'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']
    
    # Create dictionary to map class levels to never used (0) or used (1)
    class_mapping = {
        'CL0': 0, 'CL1': 0,  # Never used
        'CL2': 1, 'CL3': 1, 'CL4': 1, 'CL5': 1, 'CL6': 1  # Used
    }
    
    # Apply mapping to drug columns
    for col in drug_columns:
        if col in df.columns:
            df[col] = df[col].map(class_mapping)
    
    # Convert categorical variables to numerical
    categorical_cols = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity']
    
    # Handle specific categorical variables based on dataset documentation
    # Simplified mapping for demonstration
    age_mapping = {'18-24': 0, '25-34': 1, '35-44': 2, '45-54': 3, '55-64': 4, '65+': 5}
    gender_mapping = {'Male': 0, 'Female': 1}
    
    # Dictionary to store the mapping for each categorical column
    mapping_dict = {}
    
    for col in categorical_cols:
        if col in df.columns:
            unique_values = df[col].unique()
            mapping = {val: idx for idx, val in enumerate(unique_values)}
            df[col] = df[col].map(mapping)
            mapping_dict[col] = mapping
    
    # Store the mappings in the DataFrame metadata for later use
    df.attrs['mapping_dict'] = mapping_dict
    
    # Drop ID column as it's not needed for analysis
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)
    
    # Handle missing values
    # First, check for numerical columns
    numerical_cols = ['Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS']
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    
    # Fill missing values with mean for numerical columns
    for col in numerical_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())
    
    # Fill missing values with mode for categorical columns
    for col in categorical_cols:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Fill any remaining NaN values
    df = df.fillna(0)
    
    return df

def split_features_target(data, target_drug):
    """
    Split the dataset into features and target variable.
    
    Parameters:
    - data: DataFrame containing all data
    - target_drug: String with the name of the drug to predict
    
    Returns:
    - X: Features DataFrame
    - y: Target Series
    """
    # List of all drug columns
    drug_columns = ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 
                    'Choc', 'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 
                    'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']
    
    # Check if target drug exists in the data
    if target_drug not in data.columns:
        print(f"Warning: Target drug '{target_drug}' not found in dataset columns")
        return None, None
    
    # Create list of features (everything except drug columns, but including the target)
    features = []
    for col in data.columns:
        # Skip other drug columns, but keep the target drug
        if col in drug_columns and col != target_drug:
            continue
        features.append(col)
    
    # Create features DataFrame, excluding the target
    feature_columns = [col for col in features if col != target_drug]
    X = data[feature_columns]
    y = data[target_drug]
    
    return X, y

def prepare_train_test_data(data, target_drug, test_size=0.2, random_state=42):
    """
    Prepare training and testing datasets for modeling.
    
    Parameters:
    - data: DataFrame containing all data
    - target_drug: String with the name of the drug to predict
    - test_size: Proportion of data to use for testing
    - random_state: Random seed for reproducibility
    
    Returns:
    - X_train, X_test, y_train, y_test: Split datasets
    - scaler: Fitted StandardScaler for numerical features
    """
    X, y = split_features_target(data, target_drug)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardize numerical features
    numerical_cols = ['Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS']
    numerical_cols = [col for col in numerical_cols if col in X.columns]
    
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    return X_train, X_test, y_train, y_test, scaler

def get_user_input_features(data):
    """
    Create a form that allows users to input their personal traits for prediction.
    
    Parameters:
    - data: DataFrame with the dataset (used to get column ranges)
    
    Returns:
    - user_data: DataFrame with user input data
    """
    st.subheader("Enter Your Information")
    
    # Create columns for the form
    col1, col2 = st.columns(2)
    
    with col1:
        age_options = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
        age = st.selectbox("Age Group", options=age_options)
        
        gender = st.radio("Gender", options=["Male", "Female"])
        
        education_options = [
            "Left school before 16 years",
            "Left school at 16 years",
            "Left school at 17 years",
            "Left school at 18 years",
            "Some college or university, no certificate or degree",
            "Professional certificate/ diploma",
            "University degree",
            "Masters degree",
            "Doctorate degree"
        ]
        education = st.selectbox("Education Level", options=education_options)
        
        country_options = ["Australia", "Canada", "New Zealand", "Other", "Republic of Ireland", "UK", "USA"]
        country = st.selectbox("Country", options=country_options)
        
        ethnicity_options = ["Asian", "Black", "Mixed-Black/Asian", "Mixed-White/Asian", 
                             "Mixed-White/Black", "Other", "White"]
        ethnicity = st.selectbox("Ethnicity", options=ethnicity_options)
    
    with col2:
        # Personality traits with sliders
        nscore = st.slider("Neuroticism (N-Score)", min_value=-3.0, max_value=3.0, value=0.0, step=0.1,
                          help="Higher values indicate higher levels of neuroticism")
        
        escore = st.slider("Extraversion (E-Score)", min_value=-3.0, max_value=3.0, value=0.0, step=0.1,
                          help="Higher values indicate higher levels of extraversion")
        
        oscore = st.slider("Openness (O-Score)", min_value=-3.0, max_value=3.0, value=0.0, step=0.1,
                          help="Higher values indicate higher levels of openness to experience")
        
        ascore = st.slider("Agreeableness (A-Score)", min_value=-3.0, max_value=3.0, value=0.0, step=0.1,
                          help="Higher values indicate higher levels of agreeableness")
        
        cscore = st.slider("Conscientiousness (C-Score)", min_value=-3.0, max_value=3.0, value=0.0, step=0.1,
                          help="Higher values indicate higher levels of conscientiousness")
        
        impulsive = st.slider("Impulsiveness (Imp)", min_value=-3.0, max_value=3.0, value=0.0, step=0.1,
                             help="Higher values indicate higher levels of impulsiveness")
        
        ss = st.slider("Sensation Seeking (SS)", min_value=-3.0, max_value=3.0, value=0.0, step=0.1,
                      help="Higher values indicate higher levels of sensation seeking")
    
    # Create a dictionary with user input
    user_input = {
        'Age': age,
        'Gender': gender,
        'Education': education,
        'Country': country,
        'Ethnicity': ethnicity,
        'Nscore': nscore,
        'Escore': escore,
        'Oscore': oscore,
        'Ascore': ascore,
        'Cscore': cscore,
        'Impulsive': impulsive,
        'SS': ss
    }
    
    # Convert to DataFrame
    user_df = pd.DataFrame([user_input])
    
    # Apply the same preprocessing as the training data
    # Convert categorical variables using the same mappings
    mapping_dict = data.attrs.get('mapping_dict', {})
    
    categorical_cols = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity']
    for col in categorical_cols:
        if col in mapping_dict:
            user_df[col] = user_df[col].map(mapping_dict[col])
        else:
            # If mapping not found, use the most common value from the dataset
            user_df[col] = data[col].mode()[0]
    
    return user_df
