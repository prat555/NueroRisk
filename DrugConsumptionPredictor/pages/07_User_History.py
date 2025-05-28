import streamlit as st
import pandas as pd
import plotly.express as px
from database import get_user_history, initialize_database

# Set page title
st.set_page_config(
    page_title="User History - Drug Prediction App",
    page_icon="📜",
    layout="wide"
)

# Initialize database
if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = initialize_database()

def format_datetime(datetime_str):
    """Format datetime string for display"""
    if not datetime_str:
        return ""
    try:
        # Parse ISO format datetime
        from datetime import datetime
        dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return datetime_str

def main():
    st.title("User Prediction History")
    
    st.markdown("""
    This page shows your previous substance risk assessments. See how your predictions have changed over time
    and compare different substance risk analyses.
    """)
    
    # Get user ID from session state
    user_id = st.session_state.get('user_id')
    
    if user_id:
        # Get user history from database
        with st.spinner("Loading prediction history..."):
            history = get_user_history(user_id)
            
            if not history.empty:
                # Format the data for display
                display_history = history.copy()
                
                # Convert datetime to readable format
                if 'created_at' in display_history.columns:
                    display_history['created_at'] = display_history['created_at'].apply(format_datetime)
                
                # Format probability as percentage
                if 'probability' in display_history.columns:
                    display_history['probability'] = (display_history['probability'] * 100).round(1).astype(str) + '%'
                
                # Rename columns for display
                column_map = {
                    'id': 'ID',
                    'created_at': 'Date/Time',
                    'substance': 'Substance', 
                    'prediction': 'Predicted Usage',
                    'probability': 'Risk Probability',
                    'model_type': 'Model Used',
                    'risk_level': 'Risk Level'
                }
                
                display_history = display_history.rename(columns=column_map)
                
                # Sort by most recent first
                display_history = display_history.sort_values('Date/Time', ascending=False)
                
                # Display table
                st.subheader("Your Previous Risk Assessments")
                st.dataframe(
                    display_history[['Date/Time', 'Substance', 'Risk Probability', 'Risk Level']],
                    use_container_width=True
                )
                
                # Create visualization if we have enough data
                if len(history) > 1:
                    st.subheader("Risk Level Trends")
                    
                    # Time-based trend visualization
                    try:
                        history['created_at'] = pd.to_datetime(history['created_at'])
                        
                        fig = px.line(
                            history.sort_values('created_at'),
                            x='created_at',
                            y='probability',
                            color='substance',
                            title='Risk Probability Over Time',
                            labels={
                                'created_at': 'Date/Time',
                                'probability': 'Risk Probability',
                                'substance': 'Substance'
                            }
                        )
                        
                        fig.update_layout(yaxis_tickformat='.0%')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Substance comparison
                        if len(history['substance'].unique()) > 1:
                            fig_bar = px.bar(
                                history.groupby('substance')['probability'].mean().reset_index(),
                                x='substance',
                                y='probability',
                                color='substance',
                                title='Average Risk by Substance',
                                labels={
                                    'substance': 'Substance',
                                    'probability': 'Average Risk Probability'
                                }
                            )
                            
                            fig_bar.update_layout(yaxis_tickformat='.0%')
                            st.plotly_chart(fig_bar, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating visualizations: {e}")
            else:
                st.info("You haven't made any predictions yet. Try making a prediction first!")
    else:
        st.info("""
        No user profile found. Please create a profile by making a prediction in the main page.
        
        Your prediction history will be stored and available for review once you make your first prediction.
        """)
    
    # Add button to clear history (for future implementation)
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Clear History", disabled=True):
            # This functionality would be implemented in a future update
            st.info("This feature is coming soon!")

if __name__ == "__main__":
    main()