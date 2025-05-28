import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from database import get_substance_statistics, get_risk_level_distribution, initialize_database
import time

# Initialize database
if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = initialize_database()

def main():
    st.title("Model Analytics Dashboard")
    
    st.markdown("""
    This dashboard provides insights into the performance of the prediction models and 
    analytics based on user data and predictions stored in the database.
    """)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Prediction Statistics", "Model Performance", "User Demographics"])
    
    # Prediction Statistics tab
    with tab1:
        st.subheader("Substance Usage Predictions")
        
        # Get substance statistics from database
        with st.spinner("Loading substance statistics..."):
            substance_stats = get_substance_statistics()
            
            if not substance_stats.empty:
                # Calculate percentage of predicted usage
                substance_stats['usage_percentage'] = (
                    substance_stats['predicted_usage'] / substance_stats['total_predictions'] * 100
                ).round(1)
                
                # Display statistics
                st.dataframe(substance_stats)
                
                # Create usage distribution chart
                fig = px.bar(
                    substance_stats,
                    x='substance',
                    y=['predicted_usage', substance_stats['total_predictions'] - substance_stats['predicted_usage']],
                    labels={
                        'substance': 'Substance',
                        'value': 'Number of Predictions',
                        'variable': 'Prediction'
                    },
                    title='Predicted Usage Distribution by Substance',
                    barmode='stack',
                    color_discrete_map={
                        'predicted_usage': 'rgba(255, 99, 132, 0.8)',
                        '0': 'rgba(54, 162, 235, 0.8)'
                    }
                )
                
                fig.update_layout(
                    xaxis_title='Substance',
                    yaxis_title='Number of Predictions',
                    legend_title='Prediction',
                    height=500
                )
                
                # Update legend labels without depending on specific trace names
                fig.update_layout(
                    legend_title="Prediction Type",
                    showlegend=True
                )
                
                # Safer approach to update trace names
                for i, trace in enumerate(fig.data):
                    if "Usage" in trace.name or "1" in trace.name:
                        trace.name = "Predicted Usage"
                    else:
                        trace.name = "Predicted Non-Usage"
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Create average probability chart
                fig_prob = px.bar(
                    substance_stats,
                    x='substance',
                    y='average_probability',
                    labels={
                        'substance': 'Substance',
                        'average_probability': 'Average Probability'
                    },
                    title='Average Usage Probability by Substance',
                    color='average_probability',
                    color_continuous_scale='Viridis'
                )
                
                fig_prob.update_layout(
                    xaxis_title='Substance',
                    yaxis_title='Average Probability',
                    height=400
                )
                
                st.plotly_chart(fig_prob, use_container_width=True)
            
            else:
                st.info("No prediction data available yet. Make some predictions to see statistics here.")
        
        # Risk level distribution
        st.subheader("Risk Level Distribution")
        
        with st.spinner("Loading risk level distribution..."):
            risk_stats = get_risk_level_distribution()
            
            if not risk_stats.empty:
                # Create risk level distribution chart
                fig_risk = px.pie(
                    risk_stats,
                    values='count',
                    names='risk_level',
                    title='Distribution of Risk Levels',
                    color='risk_level',
                    color_discrete_map={
                        'low': 'rgba(75, 192, 192, 0.8)',
                        'medium': 'rgba(255, 205, 86, 0.8)',
                        'high': 'rgba(255, 99, 132, 0.8)'
                    }
                )
                
                fig_risk.update_layout(height=400)
                st.plotly_chart(fig_risk, use_container_width=True)
            else:
                st.info("No risk assessment data available yet.")
    
    # Model Performance tab
    with tab2:
        st.subheader("Model Performance Metrics")
        
        # Check for model metrics files
        model_dir = "pretrained_models"
        if os.path.exists(model_dir):
            metrics_files = [f for f in os.listdir(model_dir) if f.startswith("metrics_")]
            
            if metrics_files:
                # Read all metrics files
                all_metrics = []
                for metrics_file in metrics_files:
                    try:
                        # Extract drug and model type
                        parts = metrics_file.replace("metrics_", "").replace(".json", "").split("_")
                        model_type = parts[0]
                        drug = "_".join(parts[1:])
                        
                        # Read metrics
                        metrics = pd.read_json(os.path.join(model_dir, metrics_file), typ='series')
                        metrics['drug'] = drug
                        metrics['model_type'] = model_type
                        
                        all_metrics.append(metrics)
                    except Exception as e:
                        st.error(f"Error reading metrics file {metrics_file}: {e}")
                
                if all_metrics:
                    # Convert to DataFrame
                    metrics_df = pd.DataFrame(all_metrics)
                    
                    # Display metrics table
                    st.dataframe(
                        metrics_df[['drug', 'model_type', 'accuracy', 'precision', 'recall', 'f1', 'training_time']]
                        .sort_values(['drug', 'model_type'])
                    )
                    
                    # Create comparison chart
                    fig = px.bar(
                        metrics_df.melt(
                            id_vars=['drug', 'model_type'],
                            value_vars=['accuracy', 'precision', 'recall', 'f1'],
                            var_name='metric',
                            value_name='score'
                        ),
                        x='drug',
                        y='score',
                        color='metric',
                        facet_col='model_type',
                        labels={
                            'drug': 'Substance',
                            'score': 'Score',
                            'metric': 'Metric'
                        },
                        title='Model Performance Metrics by Substance and Model Type',
                        barmode='group'
                    )
                    
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create training time comparison
                    fig_time = px.bar(
                        metrics_df,
                        x='drug',
                        y='training_time',
                        color='model_type',
                        labels={
                            'drug': 'Substance',
                            'training_time': 'Training Time (seconds)',
                            'model_type': 'Model Type'
                        },
                        title='Model Training Time by Substance and Model Type',
                        barmode='group'
                    )
                    
                    fig_time.update_layout(height=400)
                    st.plotly_chart(fig_time, use_container_width=True)
                    
                    # Create class distribution chart
                    fig_dist = px.bar(
                        metrics_df,
                        x='drug',
                        y=['positive_samples', 'negative_samples'],
                        labels={
                            'drug': 'Substance',
                            'value': 'Number of Samples',
                            'variable': 'Class'
                        },
                        title='Class Distribution in Test Data by Substance',
                        barmode='stack'
                    )
                    
                    # Update legend labels
                    newnames = {'positive_samples': 'Users', 'negative_samples': 'Non-Users'}
                    fig_dist.for_each_trace(lambda t: t.update(name = newnames[t.name]))
                    
                    fig_dist.update_layout(height=400)
                    st.plotly_chart(fig_dist, use_container_width=True)
            else:
                st.info("No model metrics available. Train models to see performance metrics.")
        else:
            st.warning("Model directory not found. Please train models first.")
    
    # User Demographics tab
    with tab3:
        st.subheader("User Demographics (Coming Soon)")
        
        st.info("""
        This section will provide insights into the demographics of users who have made predictions,
        including age distribution, education levels, and personality trait distributions.
        """)
        
        # Placeholder for future demographic charts
        demo_placeholder = st.empty()
        demo_placeholder.markdown("""
        Future visualizations will include:
        - Age distribution of users
        - Education level breakdown
        - Personality trait distributions
        - Correlations between demographics and substance use predictions
        """)

if __name__ == "__main__":
    main()