import pandas as pd
import streamlit as st
import base64
import io
import matplotlib.pyplot as plt
from datetime import datetime
import json
import plotly.io as pio

def apply_custom_css():
    """
    Apply custom CSS styling across the application to ensure consistent look and feel.
    This includes fixing capitalization in menu items.
    """
    return st.markdown("""
    <style>
        /* Fix capitalization for all menu items that contain 'app' */
        span:not([style*="display: none"]):contains("app") {
            text-transform: capitalize !important;
        }
        
        /* Apply to all sidebar navigation elements */
        nav span:contains("app") {
            text-transform: capitalize !important;
        }
        
        /* Apply to page titles */
        h1:contains("app") {
            text-transform: capitalize !important;
        }
    </style>
    """, unsafe_allow_html=True)

def download_report(content, filename="report.html"):
    """
    Create a download button for a report.
    
    Parameters:
    - content: String with HTML content
    - filename: String with the filename
    """
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}">Download Report</a>'
    return href

def generate_report(user_data, prediction_results, risk_profile, model_info=None, figures=None):
    """
    Generate an HTML report with prediction results and insights.
    
    Parameters:
    - user_data: DataFrame with user input
    - prediction_results: Dict with prediction results
    - risk_profile: Dict with generated risk profile
    - model_info: Dict with model information
    - figures: List of Plotly figures to include
    
    Returns:
    - report: String with HTML content
    """
    # Start HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Drug Consumption Risk Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 20px;
                color: #333;
            }}
            .header {{
                background-color: #5b9bd5;
                color: white;
                padding: 20px;
                text-align: center;
            }}
            .section {{
                margin: 20px 0;
                padding: 15px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            .footer {{
                margin-top: 30px;
                text-align: center;
                font-size: 12px;
                color: #777;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                padding: 8px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            .risk-low {{
                color: green;
                font-weight: bold;
            }}
            .risk-medium {{
                color: orange;
                font-weight: bold;
            }}
            .risk-high {{
                color: red;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Drug Consumption Risk Assessment Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div>
        
        <div class="section">
            <h2>User Profile</h2>
            <table>
                <tr>
                    <th>Characteristic</th>
                    <th>Value</th>
                </tr>
    """
    
    # Add user data to table
    user_dict = user_data.iloc[0].to_dict()
    for key, value in user_dict.items():
        html_content += f"""
                <tr>
                    <td>{key}</td>
                    <td>{value}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
        
        <div class="section">
            <h2>Prediction Results</h2>
    """
    
    # Add prediction results
    if prediction_results:
        html_content += "<table><tr><th>Substance</th><th>Prediction</th><th>Probability</th></tr>"
        
        for drug, result in prediction_results.items():
            prediction = result.get('prediction', 'N/A')
            probability = result.get('probability', 'N/A')
            
            if prediction == 1:
                prediction_text = "Likely to Use"
            else:
                prediction_text = "Unlikely to Use"
            
            if isinstance(probability, (int, float)):
                probability_text = f"{probability:.2%}"
            else:
                probability_text = "N/A"
            
            html_content += f"""
                <tr>
                    <td>{drug}</td>
                    <td>{prediction_text}</td>
                    <td>{probability_text}</td>
                </tr>
            """
        
        html_content += "</table>"
    else:
        html_content += "<p>No prediction results available.</p>"
    
    html_content += """
        </div>
        
        <div class="section">
            <h2>Risk Assessment</h2>
    """
    
    # Add risk profile - simplified for data-driven approach
    if risk_profile:
        risk_level = risk_profile.get("risk_level", "unknown")
        risk_score = risk_profile.get("overall_risk_score", 0.0)
        risk_class = f"risk-{risk_level.lower()}" if risk_level.lower() in ["low", "medium", "high"] else ""
        
        html_content += f"""
            <h3>Overall Risk Level: <span class="{risk_class}">{risk_level.upper()}</span></h3>
            <p>Risk Score: {risk_score:.2%}</p>
        """
        
        # Add risk explanation based on level
        if risk_level == "low":
            html_content += """
            <p>Based on the model predictions, your profile indicates a relatively low risk for substance use.
            This suggests your demographic and personality factors align more with non-users than users
            in our reference dataset.</p>
            """
        elif risk_level == "medium":
            html_content += """
            <p>Based on the model predictions, your profile indicates a moderate risk for substance use.
            Some factors in your profile are associated with occasional substance use in our reference dataset.</p>
            """
        else:
            html_content += """
            <p>Based on the model predictions, your profile indicates a higher risk for substance use.
            Several factors in your profile are statistically associated with substance use patterns
            in our reference dataset.</p>
            """
        
        # Add substance-specific information
        html_content += """
            <h3>Substance Risk Assessment</h3>
            <table>
                <tr>
                    <th>Substance</th>
                    <th>Risk Level</th>
                </tr>
        """
        
        # Add substance risk levels based on prediction results
        for drug, result in prediction_results.items():
            probability = result.get('probability', 0.0)
            if probability > 0.7:
                risk_text = "High"
                risk_class = "risk-high"
            elif probability > 0.3:
                risk_text = "Medium"
                risk_class = "risk-medium"
            else:
                risk_text = "Low"
                risk_class = "risk-low"
                
            html_content += f"""
                <tr>
                    <td>{drug}</td>
                    <td class="{risk_class}">{risk_text} ({probability:.2%})</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h3>Risk Factors</h3>
            <p>The following factors are generally associated with higher substance use risk:</p>
            <ul>
                <li>High sensation seeking (SS) score</li>
                <li>High impulsivity score</li>
                <li>High openness to experience (Oscore)</li>
                <li>Age group (varies by substance)</li>
            </ul>
            
            <h3>Protective Factors</h3>
            <p>The following factors are generally associated with lower substance use risk:</p>
            <ul>
                <li>High conscientiousness (Cscore)</li>
                <li>High agreeableness (Ascore)</li>
                <li>More stable education and career patterns</li>
            </ul>
        """
    else:
        html_content += "<p>No risk assessment available.</p>"
    
    html_content += """
        </div>
    """
    
    # Add model information if available
    if model_info:
        html_content += """
        <div class="section">
            <h2>Model Information</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
        """
        
        for key, value in model_info.items():
            if isinstance(value, (int, float)):
                value_text = f"{value:.4f}"
            else:
                value_text = str(value)
            
            html_content += f"""
                <tr>
                    <td>{key}</td>
                    <td>{value_text}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </div>
        """
    
    # Add figures if available
    if figures:
        html_content += """
        <div class="section">
            <h2>Visualizations</h2>
        """
        
        for i, fig in enumerate(figures):
            if fig:
                try:
                    # Convert Plotly figure to HTML
                    fig_html = pio.to_html(fig, full_html=False)
                    html_content += f"""
                    <div class="figure">
                        {fig_html}
                    </div>
                    <br>
                    """
                except Exception as e:
                    html_content += f"<p>Error generating visualization {i+1}: {str(e)}</p>"
        
        html_content += """
        </div>
        """
    
    # Add disclaimer and footer
    html_content += """
        <div class="section">
            <h2>Disclaimer</h2>
            <p>This assessment is based on statistical models and general research findings. 
            It is not a clinical diagnosis and should not replace professional medical or psychological advice.
            The information is provided for educational purposes only.</p>
        </div>
        
        <div class="footer">
            <p>Generated by Drug Consumption Prediction Platform</p>
        </div>
    </body>
    </html>
    """
    
    return html_content

def create_downloadable_report(user_data, prediction_results, risk_profile, model_info=None, figures=None):
    """
    Create a downloadable report button.
    
    Parameters:
    - user_data: DataFrame with user input
    - prediction_results: Dict with prediction results
    - risk_profile: Dict with generated risk profile
    - model_info: Dict with model information
    - figures: List of Plotly figures to include
    """
    report_html = generate_report(user_data, prediction_results, risk_profile, model_info, figures)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"drug_risk_report_{timestamp}.html"
    
    # Create download button
    st.markdown(download_report(report_html, filename), unsafe_allow_html=True)

def export_model_results(model_name, metrics, feature_importance=None):
    """
    Export model results to JSON.
    
    Parameters:
    - model_name: String with the name of the model
    - metrics: Dict with model metrics
    - feature_importance: DataFrame with feature importance
    
    Returns:
    - json_str: String with JSON representation of results
    """
    result = {
        "model_name": model_name,
        "metrics": metrics
    }
    
    if feature_importance is not None and not feature_importance.empty:
        result["feature_importance"] = feature_importance.to_dict(orient="records")
    
    return json.dumps(result, indent=2)

def get_css():
    """
    Define custom CSS styles for the app.
    """
    return """
    <style>
    .reportview-container .main {
        padding-top: 0rem;
    }
    .reportview-container .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #5b9bd5;
    }
    .stAlert > div {
        padding: 0.5rem;
        border-radius: 0.3rem;
    }
    .st-emotion-cache-16txtl3 h1, .st-emotion-cache-16txtl3 h2, .st-emotion-cache-16txtl3 h3 {
        padding-top: 1rem;
        padding-bottom: 0.5rem;
    }
    </style>
    """
