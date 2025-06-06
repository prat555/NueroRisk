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
├── app.py                  # Streamlit application  
├── models/                 # Trained ML models  
│   ├── cannabis_model.pkl  
│   ├── alcohol_model.pkl  
│   └── ...  
├── data/                   # Datasets  
│   ├── raw/  
│   └── processed/  
├── notebooks/              # Jupyter notebooks for EDA & training  
├── config/                 # Database & API configs  
│   └── db_config.py  
└── requirements.txt        # Dependencies  
```  

## 🔮 Future Enhancements  

- **Mobile App Integration**  
- **Wearable Device Compatibility** (e.g., Fitbit, Apple Health)  
- **Federated Learning** for privacy-preserving predictions  
-  **Multi-language Support**  
