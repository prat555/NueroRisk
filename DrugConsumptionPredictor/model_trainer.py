import os
import pandas as pd
import numpy as np
import time
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_processor import load_data, preprocess_data, split_features_target
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = "pretrained_models"
TARGET_DRUGS = [
    'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 
    'Choc', 'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 
    'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA'
]
PRIORITY_DRUGS = ['Alcohol', 'Cannabis', 'Nicotine', 'Cocaine', 'Ecstasy', 'Mushrooms']

def ensure_model_directory():
    """Ensure the model directory exists"""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        logger.info(f"Created model directory: {MODEL_DIR}")

def train_and_save_model(data, target_drug, model_type='RandomForest'):
    """
    Train a model and save it to disk for later use
    
    Parameters:
    - data: DataFrame with the prepared dataset
    - target_drug: String with the name of the drug to predict
    - model_type: String with the type of model to train
    
    Returns:
    - model_path: Path to the saved model
    - metrics: Dict with model performance metrics
    """
    logger.info(f"Training model for {target_drug} using {model_type}...")
    
    try:
        # Split features and target
        X, y = split_features_target(data, target_drug)
        
        # Handle case where drug column doesn't exist
        if X is None or y is None:
            logger.warning(f"Could not train model for {target_drug}: column not found in dataset")
            return None, None
        
        # Split into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Standardize numerical features
        numerical_cols = ['Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS']
        numerical_cols = [col for col in numerical_cols if col in X.columns]
        
        scaler = StandardScaler()
        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
        
        # Select model type
        if model_type == 'RandomForest':
            model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                n_jobs=-1  # Use all available cores
            )
        elif model_type == 'GradientBoosting':
            model = GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            )
        else:
            logger.warning(f"Unknown model type: {model_type}, using RandomForest")
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Train the model
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'training_time': training_time,
            'test_samples': len(y_test),
            'positive_samples': sum(y_test),
            'negative_samples': len(y_test) - sum(y_test),
        }
        
        # Save model and scaler
        ensure_model_directory()
        model_filename = f"{model_type}_{target_drug}.joblib"
        scaler_filename = f"scaler_{target_drug}.joblib"
        model_path = os.path.join(MODEL_DIR, model_filename)
        scaler_path = os.path.join(MODEL_DIR, scaler_filename)
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        # Save metrics
        metrics_filename = f"metrics_{model_type}_{target_drug}.json"
        metrics_path = os.path.join(MODEL_DIR, metrics_filename)
        pd.Series(metrics).to_json(metrics_path)
        
        logger.info(f"Model for {target_drug} trained and saved successfully")
        logger.info(f"Metrics: {metrics}")
        
        return model_path, metrics
    
    except Exception as e:
        logger.error(f"Error training model for {target_drug}: {e}")
        return None, None

def load_pretrained_model(target_drug, model_type='RandomForest'):
    """
    Load a pretrained model from disk
    
    Parameters:
    - target_drug: String with the name of the drug to predict
    - model_type: String with the type of model to load
    
    Returns:
    - model: Trained model
    - scaler: StandardScaler fitted on training data
    """
    try:
        # Build paths
        model_filename = f"{model_type}_{target_drug}.joblib"
        scaler_filename = f"scaler_{target_drug}.joblib"
        model_path = os.path.join(MODEL_DIR, model_filename)
        scaler_path = os.path.join(MODEL_DIR, scaler_filename)
        
        # Check if files exist
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            logger.warning(f"Pretrained model or scaler for {target_drug} not found")
            return None, None
        
        # Load model and scaler
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        logger.info(f"Loaded pretrained model and scaler for {target_drug}")
        return model, scaler
    
    except Exception as e:
        logger.error(f"Error loading pretrained model for {target_drug}: {e}")
        return None, None

def train_all_models(priority_only=True):
    """
    Train models for all target drugs and save them to disk
    
    Parameters:
    - priority_only: If True, only train models for priority drugs
    
    Returns:
    - results: Dict with training results
    """
    results = {}
    
    try:
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        raw_data = load_data()
        if raw_data is None:
            logger.error("Failed to load data")
            return results
        
        processed_data = preprocess_data(raw_data)
        if processed_data is None:
            logger.error("Failed to preprocess data")
            return results
        
        # Select drugs to train models for
        drugs_to_train = PRIORITY_DRUGS if priority_only else TARGET_DRUGS
        
        # Train models for each drug
        for drug in drugs_to_train:
            if drug in processed_data.columns:
                # Train RandomForest (default)
                model_path, metrics = train_and_save_model(processed_data, drug)
                results[drug] = {
                    'model_path': model_path,
                    'metrics': metrics
                }
                
                # For priority drugs, also train GradientBoosting
                if drug in PRIORITY_DRUGS:
                    gb_model_path, gb_metrics = train_and_save_model(
                        processed_data, drug, model_type='GradientBoosting'
                    )
                    results[f"{drug}_GB"] = {
                        'model_path': gb_model_path,
                        'metrics': gb_metrics
                    }
            else:
                logger.warning(f"Drug {drug} not found in dataset columns")
        
        logger.info(f"Trained models for {len(results)} drug-model combinations")
        return results
    
    except Exception as e:
        logger.error(f"Error training all models: {e}")
        return results

def get_best_model_type(target_drug):
    """
    Determine the best model type for a target drug based on saved metrics
    
    Parameters:
    - target_drug: String with the name of the drug to predict
    
    Returns:
    - best_model_type: String with the best model type
    """
    try:
        # Check for RandomForest metrics
        rf_metrics_path = os.path.join(MODEL_DIR, f"metrics_RandomForest_{target_drug}.json")
        gb_metrics_path = os.path.join(MODEL_DIR, f"metrics_GradientBoosting_{target_drug}.json")
        
        rf_exists = os.path.exists(rf_metrics_path)
        gb_exists = os.path.exists(gb_metrics_path)
        
        # If only one model type exists, return it
        if rf_exists and not gb_exists:
            return 'RandomForest'
        if gb_exists and not rf_exists:
            return 'GradientBoosting'
        
        # If both exist, compare F1 scores
        if rf_exists and gb_exists:
            rf_metrics = pd.read_json(rf_metrics_path, typ='series')
            gb_metrics = pd.read_json(gb_metrics_path, typ='series')
            
            if gb_metrics['f1'] > rf_metrics['f1']:
                return 'GradientBoosting'
            else:
                return 'RandomForest'
        
        # Default to RandomForest if no metrics found
        return 'RandomForest'
    
    except Exception as e:
        logger.error(f"Error determining best model type for {target_drug}: {e}")
        return 'RandomForest'

if __name__ == "__main__":
    # This will run when the script is executed directly
    logger.info("Starting model training process...")
    train_all_models(priority_only=False)
    logger.info("Model training complete")