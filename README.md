# Drug Consumption Risk Prediction Platform  

A web-based platform that predicts an individual's risk of consuming substances (Cannabis, Alcohol, Nicotine, Ecstasy, and Mushrooms) based on psychological and demographic profiles. Built with **Python**, **Streamlit**, and **PostgreSQL**, it leverages machine learning to provide real-time risk assessments for early intervention in addiction prevention.  

## ✨ Features  

- **Personalized Risk Prediction** – Analyzes personality traits (Big Five, impulsivity, sensation-seeking) to predict substance use risk.  
- **Multiple ML Models** – Random Forest, XGBoost, and Logistic Regression with **75-85% accuracy**.  
- **Interactive Dashboard** – User-friendly interface with real-time visualizations.  
- **Persistent Storage** – PostgreSQL database for tracking predictions over time.  
- **Analytics & Insights** – Visualizes trait-risk correlations for better understanding.  

## 📦 Installation  

### Prerequisites  
- Python 3.8+  
- PostgreSQL (for database storage)  

### Steps  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/yourusername/drug-consumption-risk-prediction.git  
   cd drug-consumption-risk-prediction  
   ```  

2. Set up a virtual environment:  
   ```bash  
   python -m venv venv  
   source venv/bin/activate  # Linux/Mac  
   venv\Scripts\activate     # Windows  
   ```  

3. Install dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  

4. Configure PostgreSQL:  
   - Create a database and update `config/db_config.py` with your credentials.  

5. Run the Streamlit app:  
   ```bash  
   streamlit run app.py  
   ```  

## 🛠️ Tech Stack  

- **Frontend**: Streamlit  
- **Backend**: Python (Scikit-learn, Pandas, NumPy)  
- **Database**: PostgreSQL  
- **ML Models**: Random Forest, XGBoost, Logistic Regression  

## 📊 Model Performance  

| Model               | Accuracy | Precision | Recall | F1-Score |  
|---------------------|----------|-----------|--------|----------|  
| **Random Forest**   | 85%      | 0.84      | 0.83   | 0.83     |  
| **XGBoost**         | 82%      | 0.81      | 0.80   | 0.80     |  
| **Logistic Regression** | 75%  | 0.74      | 0.73   | 0.73     |  

## 📂 Project Structure  

```  
drug-consumption-risk-prediction/  
├── **app.py**                 # Main Streamlit application entry point  
├── **app_startup.py**         # Initialization scripts (e.g., DB connections)  
├── **app_startup.log**        # Logs from application startup  
│  
├── **data_processor.py**      # Data cleaning/feature engineering  
├── **model_trainer.py**       # ML model training pipelines  
├── **ml_models.py**           # Model definitions (Random Forest, XGBoost, etc.)  
├── **initialize_models.py**   # Loads pretrained models  
├── **model_initialization.leg** # Legacy model config (if applicable)  
│  
├── **databases.py**           # PostgreSQL interaction logic  
├── **utils.py**               # Helper functions (e.g., logging, calculations)  
│  
├── **create_figures.py**      # Generates visualizations (e.g., trait-risk plots)  
├── **create_tables.py**       # Builds summary tables for analytics  
│  
├── **/pretrained_models**     # Saved model binaries (.pkl, .h5, etc.)  
├── **/project_figures**       # Exported charts (PNG/SVG) for reports  
├── **/project_tables**        # Exported data tables (CSV/Excel)  
├── **/attached_assets**       # Miscellaneous files (e.g., icons, docs)  
│  
├── **geniu_imights.py**       # [Note: Typo? Likely "genius_insights.py"]  
├── **generated-icon.png**     # App icon/favicon  
├── **ppproject.toml**        # Project config (dependencies, metadata)  
├── **no.keck**               # [Note: Unclear purpose—verify or remove]  
│  
└── **Drug_Consumption_Prediction_Project_Report.md**  # Detailed project documentation  
```  

## 🔮 Future Enhancements  

- **Mobile App Integration**  
- **Wearable Device Compatibility** (e.g., Fitbit, Apple Health)  
- **Federated Learning** for privacy-preserving predictions  
-  **Multi-language Support**  
