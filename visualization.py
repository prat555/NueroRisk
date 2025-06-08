import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

def plot_drug_usage_distribution(data):
    """
    Plot the distribution of drug usage across the dataset.
    
    Parameters:
    - data: DataFrame with the dataset
    
    Returns:
    - fig: Plotly figure
    """
    # List of all drug columns
    drug_columns = ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 
                    'Choc', 'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 
                    'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']
    
    # Calculate usage percentage for each drug
    usage_data = []
    for drug in drug_columns:
        if drug in data.columns:
            usage_percentage = data[drug].mean() * 100  # Mean of binary values gives percentage
            usage_data.append({
                'Drug': drug,
                'Usage (%)': usage_percentage
            })
    
    # Convert to DataFrame and sort
    usage_df = pd.DataFrame(usage_data)
    usage_df = usage_df.sort_values('Usage (%)', ascending=False)
    
    # Create bar chart
    fig = px.bar(
        usage_df,
        x='Drug',
        y='Usage (%)',
        title='Drug Usage Distribution',
        labels={'Drug': 'Substance', 'Usage (%)': 'Percentage of Users'},
        color='Usage (%)',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        xaxis_title="Substance",
        yaxis_title="Percentage of Users (%)",
        xaxis={'categoryorder': 'total descending'},
        height=500
    )
    
    return fig

def plot_personality_drug_relationship(data, drug, trait):
    """
    Plot the relationship between a personality trait and drug usage.
    
    Parameters:
    - data: DataFrame with the dataset
    - drug: String with the name of the drug
    - trait: String with the name of the personality trait
    
    Returns:
    - fig: Plotly figure
    """
    # Create violin plot
    fig = px.violin(
        data,
        y=trait,
        x=drug,
        box=True,
        points="all",
        color=drug,
        labels={drug: 'Drug Usage', trait: trait},
        title=f'Relationship between {trait} and {drug} Usage',
        category_orders={drug: [0, 1]},
        color_discrete_map={0: '#636EFA', 1: '#EF553B'},
        height=500
    )
    
    # Update x-axis labels
    fig.update_xaxes(ticktext=["Not Used", "Used"], tickvals=[0, 1])
    
    return fig

def plot_correlation_heatmap(data, target_drug=None):
    """
    Plot a correlation heatmap for numerical features.
    
    Parameters:
    - data: DataFrame with the dataset
    - target_drug: Optional string with the name of the drug to highlight
    
    Returns:
    - fig: Plotly figure
    """
    # Select only numerical columns
    numerical_cols = ['Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS']
    
    # Add target drug if provided
    if target_drug and target_drug in data.columns:
        numerical_cols.append(target_drug)
    
    # Filter columns that exist in the dataframe
    numerical_cols = [col for col in numerical_cols if col in data.columns]
    
    # Calculate correlation matrix
    corr_matrix = data[numerical_cols].corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title='Correlation Heatmap for Personality Traits',
        height=600,
        width=700
    )
    
    fig.update_layout(
        xaxis_title="Features",
        yaxis_title="Features"
    )
    
    return fig

def plot_demographic_drug_usage(data, demographic_col, drug_col):
    """
    Plot drug usage distribution by demographic category.
    
    Parameters:
    - data: DataFrame with the dataset
    - demographic_col: String with the name of the demographic column
    - drug_col: String with the name of the drug column
    
    Returns:
    - fig: Plotly figure
    """
    # Calculate usage percentage by demographic group
    if demographic_col not in data.columns or drug_col not in data.columns:
        st.error(f"Columns {demographic_col} or {drug_col} not found in the dataset.")
        return None
    
    # Group by demographic and calculate mean (percentage) of drug usage
    grouped_data = data.groupby(demographic_col)[drug_col].mean().reset_index()
    grouped_data[f'{drug_col} Usage (%)'] = grouped_data[drug_col] * 100
    
    # Create bar chart
    fig = px.bar(
        grouped_data,
        x=demographic_col,
        y=f'{drug_col} Usage (%)',
        title=f'{drug_col} Usage by {demographic_col}',
        labels={demographic_col: demographic_col, f'{drug_col} Usage (%)': 'Percentage of Users (%)'},
        color=f'{drug_col} Usage (%)',
        color_continuous_scale='viridis',
        height=500
    )
    
    return fig

def plot_feature_importance(feature_importance, top_n=10):
    """
    Plot feature importance from a trained model.
    
    Parameters:
    - feature_importance: DataFrame with feature importance
    - top_n: Number of top features to show
    
    Returns:
    - fig: Plotly figure
    """
    if feature_importance is None or len(feature_importance) == 0:
        st.warning("No feature importance data available for this model.")
        return None
    
    # Get top N features
    top_features = feature_importance.head(top_n)
    
    # Create bar chart
    fig = px.bar(
        top_features,
        x='importance',
        y='feature',
        orientation='h',
        title=f'Top {top_n} Feature Importance',
        labels={'importance': 'Importance Score', 'feature': 'Feature'},
        color='importance',
        color_continuous_scale='viridis',
        height=500
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def plot_age_drug_usage(data):
    """
    Plot drug usage patterns by age group.
    
    Parameters:
    - data: DataFrame with the dataset
    
    Returns:
    - fig: Plotly figure
    """
    # List of all drug columns
    drug_columns = ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 
                    'Choc', 'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 
                    'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']
    
    # Filter columns that exist in the dataframe
    drug_columns = [col for col in drug_columns if col in data.columns]
    
    # Group by age and calculate mean for each drug
    if 'Age' not in data.columns:
        st.error("Age column not found in the dataset.")
        return None
    
    # Calculate usage percentage by age group for each drug
    age_drug_data = data.groupby('Age')[drug_columns].mean().reset_index()
    
    # Melt the dataframe for easier plotting
    age_drug_melted = age_drug_data.melt(
        id_vars=['Age'],
        value_vars=drug_columns,
        var_name='Drug',
        value_name='Usage Rate'
    )
    
    # Create line chart
    fig = px.line(
        age_drug_melted,
        x='Age',
        y='Usage Rate',
        color='Drug',
        title='Drug Usage by Age Group',
        labels={'Age': 'Age Group', 'Usage Rate': 'Usage Rate (0-1)', 'Drug': 'Substance'},
        height=600
    )
    
    # Update layout for better readability
    fig.update_layout(
        xaxis_title="Age Group",
        yaxis_title="Usage Rate (0-1)",
        legend_title="Substance",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_dashboard(data):
    """
    Create a dashboard with multiple visualizations.
    
    Parameters:
    - data: DataFrame with the dataset
    
    Returns:
    - None (displays visualizations directly)
    """
    st.title("Drug Consumption Data Dashboard")
    
    st.markdown("""
    This dashboard provides insights into drug consumption patterns based on demographic 
    information and personality traits. Use the sidebar to select different visualizations.
    """)
    
    # Create tabs for different visualization categories
    tab1, tab2, tab3 = st.tabs(["Drug Usage", "Personality Traits", "Demographics"])
    
    with tab1:
        st.subheader("Drug Usage Patterns")
        
        # Plot drug usage distribution
        fig1 = plot_drug_usage_distribution(data)
        st.plotly_chart(fig1, use_container_width=True, key="drug_usage_distribution")
        
        # Select drugs for further exploration
        drug_columns = ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 
                        'Choc', 'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 
                        'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']
        
        # Filter columns that exist in the dataframe
        drug_columns = [col for col in drug_columns if col in data.columns]
        
        selected_drug = st.selectbox("Select a drug for detailed analysis", drug_columns)
        
        # Plot age vs drug usage
        fig_age = plot_demographic_drug_usage(data, 'Age', selected_drug)
        if fig_age:
            st.plotly_chart(fig_age, use_container_width=True, key=f"tab1_age_{selected_drug}")
    
    with tab2:
        st.subheader("Personality Traits and Drug Usage")
        
        # Select drug and personality trait
        col1, col2 = st.columns(2)
        
        with col1:
            drug_columns = ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 
                            'Choc', 'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 
                            'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']
            
            # Filter columns that exist in the dataframe
            drug_columns = [col for col in drug_columns if col in data.columns]
            
            selected_drug = st.selectbox("Select a drug", drug_columns)
        
        with col2:
            personality_traits = ['Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS']
            
            # Filter columns that exist in the dataframe
            personality_traits = [col for col in personality_traits if col in data.columns]
            
            selected_trait = st.selectbox("Select a personality trait", personality_traits)
        
        # Plot personality trait vs drug usage
        fig3 = plot_personality_drug_relationship(data, selected_drug, selected_trait)
        st.plotly_chart(fig3, use_container_width=True, key=f"trait_drug_{selected_trait}_{selected_drug}")
        
        # Plot correlation heatmap
        fig4 = plot_correlation_heatmap(data, selected_drug)
        st.plotly_chart(fig4, use_container_width=True, key=f"heatmap_{selected_drug}")
    
    with tab3:
        st.subheader("Demographics and Drug Usage")
        
        # Select demographic variable and drug
        col1, col2 = st.columns(2)
        
        with col1:
            demographic_cols = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity']
            
            # Filter columns that exist in the dataframe
            demographic_cols = [col for col in demographic_cols if col in data.columns]
            
            selected_demographic = st.selectbox("Select a demographic variable", demographic_cols)
        
        with col2:
            drug_columns = ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 
                            'Choc', 'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 
                            'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']
            
            # Filter columns that exist in the dataframe
            drug_columns = [col for col in drug_columns if col in data.columns]
            
            selected_drug = st.selectbox("Select a drug type", drug_columns)
        
        # Plot demographic vs drug usage
        fig5 = plot_demographic_drug_usage(data, selected_demographic, selected_drug)
        if fig5:
            st.plotly_chart(fig5, use_container_width=True, key=f"demo_drug_{selected_demographic}_{selected_drug}")
        
        # Plot age vs drug usage patterns
        fig6 = plot_age_drug_usage(data)
        if fig6:
            st.plotly_chart(fig6, use_container_width=True, key="age_drug_usage")
