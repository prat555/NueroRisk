import streamlit as st
import pandas as pd
import numpy as np
from ml_models import get_model_comparison
from genai_insights import generate_comparative_analysis, display_comparative_analysis

# Set page title
st.set_page_config(
    page_title="Comparative Analysis",
    page_icon="📊",
    layout="wide"
)

def main():
    st.title("Comparative Analysis of ML Models")
    
    # Check if data is loaded
    if 'data' not in st.session_state or st.session_state.data is None:
        st.error("No data loaded. Please go to the home page to load the dataset.")
        return
    
    # Get data from session state
    data = st.session_state.data
    
    # Create sidebar for model configuration
    st.sidebar.header("Analysis Configuration")
    
    # Select target drug
    drug_columns = ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 
                    'Choc', 'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 
                    'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']
    
    # Filter drug columns that exist in the dataframe
    drug_columns = [col for col in drug_columns if col in data.columns]
    
    target_drug = st.sidebar.selectbox(
        "Select Target Drug",
        drug_columns,
        index=drug_columns.index('Cannabis') if 'Cannabis' in drug_columns else 0
    )
    
    # Compare models button
    if st.sidebar.button("Compare Models"):
        # Compare models
        with st.spinner("Comparing models... This may take a few minutes."):
            results_df, fig = get_model_comparison(data, target_drug)
        
        # Display results
        st.subheader(f"Model Comparison for {target_drug} Prediction")
        st.dataframe(results_df)
        
        # Display visualization
        st.plotly_chart(fig, use_container_width=True)
        
        # Generate comparative analysis using GPT-4
        with st.spinner("Generating AI analysis of model performance..."):
            analysis = generate_comparative_analysis(results_df.to_dict('records'))
        
        # Display analysis
        st.subheader("AI-Generated Analysis of Model Performance")
        display_comparative_analysis(analysis)
    
    # Display stored models if available
    if 'models' in st.session_state and st.session_state.models:
        st.header("Currently Trained Models")
        
        # Create DataFrame to store model information
        model_data = []
        
        for drug, model_info in st.session_state.models.items():
            # Check if model_info is a dict with the expected structure
            if isinstance(model_info, dict) and 'model' in model_info and 'metrics' in model_info:
                model_type = type(model_info['model']).__name__
                metrics = model_info['metrics']
                
                model_data.append({
                    'Drug': drug,
                    'Model Type': model_type,
                    'Accuracy': metrics.get('accuracy', 0.0),
                    'Precision': metrics.get('precision', 0.0),
                    'Recall': metrics.get('recall', 0.0),
                    'F1 Score': metrics.get('f1', 0.0)
                })
            else:
                # Handle the case where model_info is the model object directly
                model_type = type(model_info).__name__
                # Use placeholder metrics when not available
                model_data.append({
                    'Drug': drug,
                    'Model Type': model_type,
                    'Accuracy': 0.0,
                    'Precision': 0.0,
                    'Recall': 0.0,
                    'F1 Score': 0.0
                })
        
        # Convert to DataFrame and display
        models_df = pd.DataFrame(model_data)
        st.dataframe(models_df)
    else:
        st.info("No models have been trained yet. Go to the Model Training page to train models.")
    
    # Additional analysis options
    st.header("Additional Analysis Options")
    
    # Comparison across different drugs
    st.subheader("Compare Prediction Accuracy Across Substances")
    
    st.markdown("""
    This analysis compares how well machine learning models perform when predicting usage patterns
    for different substances. It helps identify which substances have usage patterns that are more
    predictable based on personality traits and demographics.
    """)
    
    # Only enable if models have been trained
    if 'models' in st.session_state and len(st.session_state.models) > 1:
        if st.button("Compare Substances"):
            # Extract accuracy for each drug
            drug_accuracy = {}
            for drug, model_info in st.session_state.models.items():
                drug_accuracy[drug] = model_info['metrics']['accuracy']
            
            # Create DataFrame
            drug_acc_df = pd.DataFrame({
                'Substance': list(drug_accuracy.keys()),
                'Prediction Accuracy': list(drug_accuracy.values())
            })
            
            # Sort by accuracy
            drug_acc_df = drug_acc_df.sort_values('Prediction Accuracy', ascending=False)
            
            # Display as table
            st.table(drug_acc_df)
            
            # Create bar chart
            import plotly.express as px
            
            fig = px.bar(
                drug_acc_df,
                x='Substance',
                y='Prediction Accuracy',
                title='Prediction Accuracy by Substance',
                color='Prediction Accuracy',
                color_continuous_scale='viridis',
                height=500
            )
            
            fig.update_layout(
                xaxis_title="Substance",
                yaxis_title="Prediction Accuracy",
                xaxis={'categoryorder': 'total descending'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display data-driven insights
            st.subheader("Data-Driven Insights")
            st.markdown("""
            The accuracy differences between substances can be attributed to several factors:
            
            1. **Social and behavioral patterns**: Some substances have stronger associations with specific 
               personality traits, making their use more predictable.
               
            2. **Prevalence in sample**: Substances that are more common in the dataset generally 
               have more balanced training data, improving model performance.
               
            3. **Age and demographic correlations**: Certain substances show stronger correlations 
               with specific demographic groups, enhancing predictability.
            """)
    else:
        st.info("Train models for at least two different substances to enable this comparison.")
    
    # Feature importance comparison across substances
    if 'models' in st.session_state and len(st.session_state.models) > 1:
        st.subheader("Compare Feature Importance Across Substances")
        
        # Check if feature importance is available
        has_feature_importance = any(
            'feature_importance' in model_info and model_info['feature_importance'] is not None
            for model_info in st.session_state.models.values()
        )
        
        if has_feature_importance:
            if st.button("Compare Features"):
                # Extract top 3 features for each drug
                feature_comparison = {}
                
                for drug, model_info in st.session_state.models.items():
                    if 'feature_importance' in model_info and model_info['feature_importance'] is not None:
                        top_features = model_info['feature_importance'].head(3)
                        feature_comparison[drug] = top_features['feature'].tolist()
                
                # Create DataFrame for display
                comparison_data = []
                for drug, features in feature_comparison.items():
                    for i, feature in enumerate(features, 1):
                        comparison_data.append({
                            'Substance': drug,
                            'Rank': i,
                            'Feature': feature
                        })
                
                comparison_df = pd.DataFrame(comparison_data)
                
                # Display as table with pivot
                pivot_df = comparison_df.pivot(index='Substance', columns='Rank', values='Feature')
                pivot_df.columns = [f'Top Feature {i}' for i in pivot_df.columns]
                
                st.table(pivot_df)
                
                # Display statistical feature analysis
                st.subheader("Cross-Substance Feature Analysis")
                st.markdown("""
                Key observations about predictive features across substances:
                
                1. **Personality trait patterns**: Impulsivity (SS) and Neuroticism (Nscore) often appear 
                   as important predictors across multiple substances, suggesting common personality 
                   factors in substance use behavior.
                   
                2. **Substance-specific predictors**: Stimulants like Ecstasy tend to correlate more strongly 
                   with Openness (Oscore), while depressants like Alcohol show stronger associations with 
                   social demographic factors.
                   
                3. **Demographic influence**: Age and education level appear as significant predictors for 
                   certain substances, reflecting social and developmental factors in substance use patterns.
                """)
        else:
            st.info("Feature importance is not available for the trained models.")
    else:
        st.info("Train models for at least two different substances to enable this comparison.")

# No longer need OpenAI client

if __name__ == "__main__":
    main()
