import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from openai import OpenAI
import time

# Initialize the OpenAI client
# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
def get_openai_client():
    """
    Initialize and return an OpenAI client using API key from environment variables.
    """
    # Check if we have the API key in session state first (for persistence)
    if 'openai_api_key' in st.session_state and st.session_state.openai_api_key:
        api_key = st.session_state.openai_api_key
    else:
        # Try to get from environment
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            # Store it in session state for persistence
            st.session_state.openai_api_key = api_key
    
    if not api_key:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return None
    
    # Add verbose warning to check if key works
    try:
        client = OpenAI(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")
        return None

def generate_risk_profile(user_data, prediction_results):
    """
    Generate a personalized risk profile using OpenAI's GPT model.
    
    Parameters:
    - user_data: DataFrame with user input data
    - prediction_results: Dict with ML prediction results
    
    Returns:
    - risk_profile: Dict with generated insights
    """
    client = get_openai_client()
    if not client:
        return {"error": "OpenAI client could not be initialized. Please check your API key."}
    
    # Extract user information
    user_info = user_data.iloc[0].to_dict()
    
    # Map numeric personality scores back to interpretable values if needed
    # For demonstration purposes - in a real app you'd have the actual mappings
    trait_interpretations = {
        'Nscore': {-3: "Very low neuroticism", -2: "Low neuroticism", -1: "Somewhat low neuroticism", 
                  0: "Average neuroticism", 1: "Somewhat high neuroticism", 2: "High neuroticism", 3: "Very high neuroticism"},
        'Escore': {-3: "Very low extraversion", -2: "Low extraversion", -1: "Somewhat low extraversion", 
                  0: "Average extraversion", 1: "Somewhat high extraversion", 2: "High extraversion", 3: "Very high extraversion"},
        'Oscore': {-3: "Very low openness", -2: "Low openness", -1: "Somewhat low openness", 
                  0: "Average openness", 1: "Somewhat high openness", 2: "High openness", 3: "Very high openness"},
        'Ascore': {-3: "Very low agreeableness", -2: "Low agreeableness", -1: "Somewhat low agreeableness", 
                  0: "Average agreeableness", 1: "Somewhat high agreeableness", 2: "High agreeableness", 3: "Very high agreeableness"},
        'Cscore': {-3: "Very low conscientiousness", -2: "Low conscientiousness", -1: "Somewhat low conscientiousness", 
                  0: "Average conscientiousness", 1: "Somewhat high conscientiousness", 2: "High conscientiousness", 3: "Very high conscientiousness"}
    }
    
    # Get interpretable values for each trait where possible
    personality_descriptions = {}
    for trait, mappings in trait_interpretations.items():
        if trait in user_info and isinstance(user_info[trait], (int, float)):
            # Find the closest mapping value
            closest_key = min(mappings.keys(), key=lambda x: abs(x - user_info[trait]))
            personality_descriptions[trait] = mappings[closest_key]
        else:
            personality_descriptions[trait] = f"Unknown {trait}"
    
    # Create a prompt with user information and prediction results
    prompt = f"""
    You are an expert in analyzing substance use risk factors and providing real-time personalized insights for individual users.
    
    User Profile:
    - Age group: {user_info.get('Age')}
    - Gender: {user_info.get('Gender')}
    - Education level: {user_info.get('Education')}
    - Neuroticism assessment: {personality_descriptions.get('Nscore')}
    - Extraversion assessment: {personality_descriptions.get('Escore')}
    - Openness assessment: {personality_descriptions.get('Oscore')}
    - Agreeableness assessment: {personality_descriptions.get('Ascore')}
    - Conscientiousness assessment: {personality_descriptions.get('Cscore')}
    - Impulsivity level: {user_info.get('Impulsive', 'N/A')}
    - Sensation seeking level: {user_info.get('SS', 'N/A')}
    
    Machine Learning Prediction Results:
    {json.dumps(prediction_results, indent=2)}
    
    Based on this information, provide a practical, actionable risk assessment for this specific individual, including:
    1. Overall risk profile (low, medium, high) with clear explanation
    2. Key personality and demographic factors that most influence their specific risk profile
    3. Specific substances that may be of higher concern based on their unique profile
    4. Protective factors in their profile that might reduce their personal risk
    5. Practical, personalized recommendations for this individual
    
    Format the response as a JSON object with the following structure:
    {
        "risk_level": string (low, medium, high),
        "risk_explanation": string,
        "key_factors": [string],
        "substances_of_concern": [string],
        "protective_factors": [string],
        "recommendations": [string]
    }
    """
    
    try:
        # Create a spinner to show processing
        with st.spinner("Generating personalized risk profile..."):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert in drug consumption risk analysis and prevention."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.5
            )
            
            # Extract and parse the response
            risk_profile_json = response.choices[0].message.content
            risk_profile = json.loads(risk_profile_json)
            
            return risk_profile
    except Exception as e:
        st.error(f"Error generating risk profile: {e}")
        return {"error": str(e)}

def generate_research_insights(drug_name):
    """
    Generate research-based insights about a specific drug using OpenAI's GPT model.
    
    Parameters:
    - drug_name: String with the name of the drug
    
    Returns:
    - insights: Dict with generated insights
    """
    client = get_openai_client()
    if not client:
        return {"error": "OpenAI client could not be initialized. Please check your API key."}
    
    # Create a prompt for research insights
    prompt = f"""
    Provide comprehensive, evidence-based information about {drug_name}, including:
    1. Brief description and classification
    2. Short-term and long-term effects
    3. Risk factors for problematic use
    4. Prevalence statistics globally
    5. Key research findings on personality traits associated with {drug_name} use
    
    Format the response as a JSON object with the following structure:
    {{
        "name": string,
        "description": string,
        "classification": string,
        "short_term_effects": [string],
        "long_term_effects": [string],
        "risk_factors": [string],
        "prevalence": string,
        "personality_associations": [string],
        "references": [string] (list of brief reference notes, not full citations)
    }}
    
    Ensure the information is accurate, balanced, and based on scientific evidence.
    """
    
    try:
        # Create a spinner to show processing
        with st.spinner(f"Generating research insights for {drug_name}..."):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a drug researcher with expertise in substance use patterns and effects."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            # Extract and parse the response
            insights_json = response.choices[0].message.content
            insights = json.loads(insights_json)
            
            return insights
    except Exception as e:
        st.error(f"Error generating research insights: {e}")
        return {"error": str(e)}

def generate_comparative_analysis(prediction_results):
    """
    Generate a comparative analysis of different ML model performances.
    
    Parameters:
    - prediction_results: Dict with prediction results from multiple models
    
    Returns:
    - analysis: Dict with generated analysis
    """
    client = get_openai_client()
    if not client:
        return {"error": "OpenAI client could not be initialized. Please check your API key."}
    
    # Create a prompt for comparative analysis
    prompt = f"""
    Analyze the following machine learning model performance results for drug consumption prediction:
    {json.dumps(prediction_results, indent=2)}
    
    Please provide:
    1. A comparative analysis of the different models' strengths and weaknesses
    2. Insights into which models performed best for which types of predictions
    3. Recommendations for model selection based on specific use cases
    4. Potential improvements to enhance model performance
    
    Format the response as a JSON object with the following structure:
    {{
        "comparative_analysis": string,
        "best_models": {{
            "overall": string,
            "precision_focused": string,
            "recall_focused": string
        }},
        "model_selection_recommendations": [string],
        "improvement_suggestions": [string]
    }}
    
    Base your analysis on machine learning best practices and the specific metrics provided.
    """
    
    try:
        # Create a spinner to show processing
        with st.spinner("Generating comparative analysis of ML models..."):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a machine learning expert specializing in classification models."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            # Extract and parse the response
            analysis_json = response.choices[0].message.content
            analysis = json.loads(analysis_json)
            
            return analysis
    except Exception as e:
        st.error(f"Error generating comparative analysis: {e}")
        return {"error": str(e)}

def display_risk_profile(risk_profile):
    """
    Display the generated risk profile in an organized and visually appealing way.
    
    Parameters:
    - risk_profile: Dict with generated risk profile
    """
    if "error" in risk_profile:
        st.error(f"Error: {risk_profile['error']}")
        return
    
    # Display risk level with appropriate color
    risk_level = risk_profile.get("risk_level", "unknown")
    if risk_level.lower() == "low":
        risk_color = "green"
    elif risk_level.lower() == "medium":
        risk_color = "orange"
    else:
        risk_color = "red"
    
    st.markdown(f"## Overall Risk Level: <span style='color:{risk_color}'>{risk_level.upper()}</span>", unsafe_allow_html=True)
    
    # Display risk explanation
    st.markdown("### Risk Assessment")
    st.write(risk_profile.get("risk_explanation", "No explanation provided."))
    
    # Create columns for key factors and protective factors
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Key Risk Factors")
        key_factors = risk_profile.get("key_factors", [])
        if isinstance(key_factors, list):
            for factor in key_factors:
                st.markdown(f"- {factor}")
        else:
            st.write(key_factors)
    
    with col2:
        st.markdown("### Protective Factors")
        protective_factors = risk_profile.get("protective_factors", [])
        if isinstance(protective_factors, list):
            for factor in protective_factors:
                st.markdown(f"- {factor}")
        else:
            st.write(protective_factors)
    
    # Display substances of concern
    st.markdown("### Substances of Potential Concern")
    substances = risk_profile.get("substances_of_concern", [])
    if isinstance(substances, list):
        for substance in substances:
            st.markdown(f"- {substance}")
    else:
        st.write(substances)
    
    # Display recommendations
    st.markdown("### Recommendations")
    recommendations = risk_profile.get("recommendations", [])
    if isinstance(recommendations, list):
        for rec in recommendations:
            st.markdown(f"- {rec}")
    else:
        st.write(recommendations)
    
    # Disclaimer
    st.markdown("---")
    st.markdown("""
    **Disclaimer**: This assessment is based on statistical models and general research findings. 
    It is not a clinical diagnosis and should not replace professional medical or psychological advice.
    """)

def display_research_insights(insights):
    """
    Display the generated research insights in an organized and visually appealing way.
    
    Parameters:
    - insights: Dict with generated research insights
    """
    if "error" in insights:
        st.error(f"Error: {insights['error']}")
        return
    
    # Display drug name and description
    st.markdown(f"## {insights.get('name', 'Unknown Substance')}")
    st.markdown(f"**Classification**: {insights.get('classification', 'Not specified')}")
    st.markdown(f"### Description")
    st.write(insights.get('description', 'No description available.'))
    
    # Create columns for short-term and long-term effects
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Short-term Effects")
        effects = insights.get("short_term_effects", [])
        if isinstance(effects, list):
            for effect in effects:
                st.markdown(f"- {effect}")
        else:
            st.write(effects)
    
    with col2:
        st.markdown("### Long-term Effects")
        effects = insights.get("long_term_effects", [])
        if isinstance(effects, list):
            for effect in effects:
                st.markdown(f"- {effect}")
        else:
            st.write(effects)
    
    # Display risk factors
    st.markdown("### Risk Factors for Problematic Use")
    factors = insights.get("risk_factors", [])
    if isinstance(factors, list):
        for factor in factors:
            st.markdown(f"- {factor}")
    else:
        st.write(factors)
    
    # Display prevalence
    st.markdown("### Prevalence")
    st.write(insights.get('prevalence', 'No prevalence data available.'))
    
    # Display personality associations
    st.markdown("### Personality Trait Associations")
    st.write(insights.get('personality_associations', 'No personality association data available.'))
    
    # Display references
    st.markdown("### References")
    references = insights.get("references", [])
    if isinstance(references, list):
        for ref in references:
            st.markdown(f"- {ref}")
    else:
        st.write(references)
    
    # Disclaimer
    st.markdown("---")
    st.markdown("""
    **Disclaimer**: This information is provided for educational purposes only. 
    It is not intended to promote drug use or to replace professional medical advice.
    """)

def display_comparative_analysis(analysis):
    """
    Display the generated comparative analysis in an organized and visually appealing way.
    
    Parameters:
    - analysis: Dict with generated comparative analysis
    """
    if "error" in analysis:
        st.error(f"Error: {analysis['error']}")
        return
    
    # Display comparative analysis
    st.markdown("## Comparative Analysis of ML Models")
    st.write(analysis.get('comparative_analysis', 'No analysis available.'))
    
    # Display best models
    st.markdown("### Best Performing Models")
    best_models = analysis.get("best_models", {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Overall Best")
        st.write(best_models.get('overall', 'Not specified'))
    
    with col2:
        st.markdown("#### Best for Precision")
        st.write(best_models.get('precision_focused', 'Not specified'))
    
    with col3:
        st.markdown("#### Best for Recall")
        st.write(best_models.get('recall_focused', 'Not specified'))
    
    # Display model selection recommendations
    st.markdown("### Model Selection Recommendations")
    recommendations = analysis.get("model_selection_recommendations", [])
    if isinstance(recommendations, list):
        for rec in recommendations:
            st.markdown(f"- {rec}")
    else:
        st.write(recommendations)
    
    # Display improvement suggestions
    st.markdown("### Improvement Suggestions")
    suggestions = analysis.get("improvement_suggestions", [])
    if isinstance(suggestions, list):
        for sug in suggestions:
            st.markdown(f"- {sug}")
    else:
        st.write(suggestions)
