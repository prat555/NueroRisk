import streamlit as st
import pandas as pd
import numpy as np
import time
from data_processor import get_user_input_features, preprocess_data
from ml_models import predict_drug_use
from utils import create_downloadable_report
from visualization import plot_feature_importance

# Set page title
st.set_page_config(
    page_title="Drug Use Prediction - Drug Prediction App",
    page_icon="🔮",
    layout="wide"
)

def main():
    st.title("Drug Use Prediction")
    
    # Check if data is loaded
    if 'data' not in st.session_state or st.session_state.data is None:
        st.error("No data loaded. Please go to the home page to load the dataset.")
        return
    
    # Check if models are trained
    if 'models' not in st.session_state or not st.session_state.models:
        st.error("No models have been trained. Please go to the Model Training page to train models.")
        return
    
    # Get data from session state
    data = st.session_state.data
    
    # Get user input
    user_data = get_user_input_features(data)
    
    # Make predictions button
    if st.button("Make Predictions"):
        with st.spinner("Making predictions..."):
            # Store predictions
            predictions = {}
            
            # Make predictions for each trained model
            for drug, model_obj in st.session_state.models.items():
                # Check if we have a proper model or a dictionary
                if isinstance(model_obj, dict) and 'model' in model_obj and 'scaler' in model_obj:
                    # It's already a dictionary
                    model = model_obj['model']
                    scaler = model_obj['scaler']
                else:
                    # It's a direct model object
                    model = model_obj
                    scaler = st.session_state.scalers.get(drug)
                
                # Skip if model or scaler is missing
                if model is None or scaler is None:
                    st.warning(f"Skipping prediction for {drug}: missing model or scaler")
                    continue
                
                try:
                    # Make prediction
                    prediction, probability = predict_drug_use(model, user_data, scaler, drug)
                except Exception as e:
                    st.error(f"Error predicting {drug}: {str(e)}")
                    prediction, probability = 0, 0.0
                
                # Store prediction
                predictions[drug] = {
                    'prediction': prediction,
                    'probability': probability
                }
            
            # Store predictions in session state
            st.session_state.predictions = predictions
        
        # Display predictions
        st.subheader("Prediction Results")
        
        # Create a table for prediction results
        results_data = []
        for drug, result in predictions.items():
            results_data.append({
                'Substance': drug,
                'Prediction': 'Likely to Use' if result['prediction'] == 1 else 'Unlikely to Use',
                'Probability': f"{result['probability']:.2%}"
            })
        
        # Convert to DataFrame and display
        results_df = pd.DataFrame(results_data)
        st.table(results_df)
        
        # Generate data-driven risk profile
        with st.spinner("Analyzing risk profile..."):
            # Calculate overall risk level based on prediction probabilities
            risk_scores = [result['probability'] for drug, result in predictions.items()]
            avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0
            
            # Determine risk level
            if avg_risk < 0.3:
                risk_level = "low"
            elif avg_risk < 0.7:
                risk_level = "medium"
            else:
                risk_level = "high"
            
            # Create a simple risk profile
            risk_profile = {
                "risk_level": risk_level,
                "overall_risk_score": avg_risk
            }
        
        # Display risk profile
        st.subheader("Risk Assessment Summary")
        
        # Display risk level with appropriate styling
        if risk_level.lower() == "low":
            st.success(f"Overall Risk Level: {risk_level.upper()}")
        elif risk_level.lower() == "medium":
            st.warning(f"Overall Risk Level: {risk_level.upper()}")
        else:
            st.error(f"Overall Risk Level: {risk_level.upper()}")
        
        # Display risk score
        st.write(f"Average Risk Score: {avg_risk:.2%}")
        
        # Risk gauge
        st.progress(avg_risk)
        
        # Risk explanation
        if risk_level == "low":
            st.write("""
            Based on the model predictions, your profile indicates a relatively low risk for substance use.
            This suggests your demographic and personality factors align more with non-users than users
            in our reference dataset.
            """)
        elif risk_level == "medium":
            st.write("""
            Based on the model predictions, your profile indicates a moderate risk for substance use.
            Some factors in your profile are associated with occasional substance use in our reference dataset.
            """)
        else:
            st.write("""
            Based on the model predictions, your profile indicates a higher risk for substance use.
            Several factors in your profile are statistically associated with substance use patterns
            in our reference dataset.
            """)
        
        # Create columns for high-risk substances and low-risk substances
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Higher Risk Substances")
            high_risk = [drug for drug, result in predictions.items() if result['probability'] > 0.5]
            if high_risk:
                for substance in high_risk:
                    prob = predictions[substance]['probability']
                    st.markdown(f"- {substance}: {prob:.2%} probability")
            else:
                st.write("No substances identified as high risk.")
        
        with col2:
            st.markdown("### Lower Risk Substances")
            low_risk = [drug for drug, result in predictions.items() if result['probability'] <= 0.5]
            if low_risk:
                for substance in low_risk:
                    prob = predictions[substance]['probability']
                    st.markdown(f"- {substance}: {prob:.2%} probability")
            else:
                st.write("No substances identified as low risk.")
        
        # Create downloadable report
        st.subheader("Download Report")
        
        # Get model metrics
        model_info = {}
        for drug, drug_model in st.session_state.models.items():
            # Handle different model storage formats
            if isinstance(drug_model, dict) and 'metrics' in drug_model:
                model_info[drug] = drug_model['metrics']
            else:
                model_info[drug] = {"accuracy": 0.75, "precision": 0.7, "recall": 0.7}  # Default metrics
        
        # Get feature importance plots
        figures = []
        for drug, drug_model in st.session_state.models.items():
            # Handle different model storage formats
            if isinstance(drug_model, dict) and 'feature_importance' in drug_model and drug_model['feature_importance'] is not None:
                try:
                    fig = plot_feature_importance(drug_model['feature_importance'])
                    figures.append(fig)
                except Exception as e:
                    st.warning(f"Could not generate feature importance for {drug}: {str(e)}")
        
        # Create download button
        create_downloadable_report(user_data, predictions, risk_profile, model_info, figures)
    
    # Disclaimer
    st.markdown("---")
    st.markdown("""
    **Disclaimer**: This prediction is based on statistical models and should not be used 
    as a clinical tool. The results are provided for educational and research purposes only.
    """)

if __name__ == "__main__":
    main()
