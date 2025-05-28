import streamlit as st
import pandas as pd
import numpy as np
from visualization import create_dashboard

# Set page title
st.set_page_config(
    page_title="Data Exploration - Drug Prediction App",
    page_icon="📊",
    layout="wide"
)

# Apply custom CSS to fix menu capitalization
st.markdown("""
<style>
    /* Fix menu item capitalization */
    span:contains('app') {
        text-transform: capitalize !important;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("Data Exploration")
    
    # Check if data is loaded
    if 'data' not in st.session_state or st.session_state.data is None:
        st.error("No data loaded. Please go to the home page to load the dataset.")
        return
    
    # Get data from session state
    data = st.session_state.data
    
    # Create dashboard with data
    create_dashboard(data)
    
    # Add additional exploration options
    st.subheader("Explore Raw Data")
    
    # Number of rows to show
    num_rows = st.slider("Number of rows to display", min_value=5, max_value=100, value=10)
    
    # Display data
    st.dataframe(data.head(num_rows))
    
    # Column statistics
    st.subheader("Column Statistics")
    
    # Select a column to explore
    column = st.selectbox("Select a column", data.columns.tolist())
    
    # Display column statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Basic Statistics")
        if pd.api.types.is_numeric_dtype(data[column]):
            st.write(data[column].describe())
        else:
            # For categorical columns, show value counts
            st.write(data[column].value_counts())
    
    with col2:
        st.write("Missing Values")
        missing = data[column].isna().sum()
        st.write(f"Number of missing values: {missing}")
        st.write(f"Percentage of missing values: {missing / len(data) * 100:.2f}%")
    
    # Allow users to download the dataset
    st.subheader("Download Dataset")
    
    # Create a CSV file from the dataframe
    csv = data.to_csv(index=False)
    
    # Create a download button
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="drug_consumption_data.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()
