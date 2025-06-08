import os
import pandas as pd
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import streamlit as st

# Create SQLAlchemy base
Base = declarative_base()

# User profile model
class UserProfile(Base):
    __tablename__ = 'user_profiles'
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    age_group = Column(String(10))
    gender = Column(String(10))
    education = Column(String(100))
    country = Column(String(50))
    ethnicity = Column(String(50))
    nscore = Column(Float)
    escore = Column(Float)
    oscore = Column(Float)
    ascore = Column(Float)
    cscore = Column(Float)
    impulsive = Column(Float)
    sensation_seeking = Column(Float)
    
    # Relationship to prediction results
    predictions = relationship("PredictionResult", back_populates="user", cascade="all, delete-orphan")
    
    def to_dict(self):
        return {
            'id': self.id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'age_group': self.age_group,
            'gender': self.gender,
            'education': self.education,
            'country': self.country,
            'ethnicity': self.ethnicity,
            'nscore': self.nscore,
            'escore': self.escore,
            'oscore': self.oscore,
            'ascore': self.ascore,
            'cscore': self.cscore,
            'impulsive': self.impulsive,
            'sensation_seeking': self.sensation_seeking
        }

# Prediction result model
class PredictionResult(Base):
    __tablename__ = 'prediction_results'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('user_profiles.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    substance = Column(String(50))
    prediction = Column(Boolean)
    probability = Column(Float)
    model_type = Column(String(50))
    risk_level = Column(String(20))
    risk_factors = Column(Text)  # Stored as JSON
    
    # Relationship to user profile
    user = relationship("UserProfile", back_populates="predictions")
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'substance': self.substance,
            'prediction': self.prediction,
            'probability': self.probability,
            'model_type': self.model_type,
            'risk_level': self.risk_level,
            'risk_factors': json.loads(self.risk_factors) if self.risk_factors else None
        }

# Get database connection
def get_database_connection():
    """
    Create and return a SQLAlchemy engine for PostgreSQL connection
    """
    # Get database URL from environment variable
    database_url = os.environ.get("DATABASE_URL")
    
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")
    
    # Create engine
    engine = create_engine(database_url)
    
    return engine

def initialize_database():
    """
    Initialize the database by creating all tables if they don't exist
    """
    try:
        engine = get_database_connection()
        Base.metadata.create_all(engine)
        return True
    except Exception as e:
        st.error(f"Error initializing database: {e}")
        return False

def get_session():
    """
    Create and return a new database session
    """
    engine = get_database_connection()
    Session = sessionmaker(bind=engine)
    return Session()

def save_user_profile(user_data):
    """
    Save a user profile to the database
    
    Parameters:
    - user_data: DataFrame with user input data
    
    Returns:
    - user_id: ID of the saved user profile
    """
    try:
        # Convert DataFrame to dict
        user_dict = user_data.iloc[0].to_dict()
        
        # Map DataFrame columns to UserProfile columns
        user_profile = UserProfile(
            age_group=str(user_dict.get('Age')),
            gender=str(user_dict.get('Gender')),
            education=str(user_dict.get('Education')),
            country=str(user_dict.get('Country')),
            ethnicity=str(user_dict.get('Ethnicity')),
            nscore=float(user_dict.get('Nscore', 0)),
            escore=float(user_dict.get('Escore', 0)),
            oscore=float(user_dict.get('Oscore', 0)),
            ascore=float(user_dict.get('Ascore', 0)),
            cscore=float(user_dict.get('Cscore', 0)),
            impulsive=float(user_dict.get('Impulsive', 0)),
            sensation_seeking=float(user_dict.get('SS', 0))
        )
        
        # Save to database
        session = get_session()
        session.add(user_profile)
        session.commit()
        
        # Get the user ID
        user_id = user_profile.id
        
        session.close()
        
        return user_id
    except Exception as e:
        st.error(f"Error saving user profile: {e}")
        return None

def save_prediction_result(user_id, substance, prediction, probability, model_type, risk_profile=None):
    """
    Save a prediction result to the database
    
    Parameters:
    - user_id: ID of the user profile
    - substance: String with the name of the substance
    - prediction: Boolean prediction (True for use, False for non-use)
    - probability: Float between 0 and 1
    - model_type: String with the name of the model
    - risk_profile: Dict with the risk profile (optional)
    
    Returns:
    - prediction_id: ID of the saved prediction result
    """
    try:
        # Create risk factors JSON
        risk_factors = None
        risk_level = None
        
        if risk_profile:
            risk_level = risk_profile.get('risk_level', 'unknown')
            # Convert risk factors to JSON
            risk_factors_dict = {
                'key_factors': risk_profile.get('key_factors', []),
                'protective_factors': risk_profile.get('protective_factors', []),
                'substances_of_concern': risk_profile.get('substances_of_concern', []),
                'recommendations': risk_profile.get('recommendations', [])
            }
            risk_factors = json.dumps(risk_factors_dict)
        
        # Create prediction result
        prediction_result = PredictionResult(
            user_id=user_id,
            substance=substance,
            prediction=bool(prediction),
            probability=float(probability),
            model_type=model_type,
            risk_level=risk_level,
            risk_factors=risk_factors
        )
        
        # Save to database
        session = get_session()
        session.add(prediction_result)
        session.commit()
        
        # Get the prediction ID
        prediction_id = prediction_result.id
        
        session.close()
        
        return prediction_id
    except Exception as e:
        st.error(f"Error saving prediction result: {e}")
        return None

def get_user_history(user_id=None, limit=100):
    """
    Get user prediction history from the database
    
    Parameters:
    - user_id: ID of the user profile (optional, get all if not provided)
    - limit: Maximum number of results to return
    
    Returns:
    - history: DataFrame with prediction history
    """
    try:
        session = get_session()
        
        # Query prediction results
        if user_id:
            results = session.query(PredictionResult).filter(
                PredictionResult.user_id == user_id
            ).order_by(PredictionResult.created_at.desc()).limit(limit).all()
        else:
            results = session.query(PredictionResult).order_by(
                PredictionResult.created_at.desc()
            ).limit(limit).all()
        
        # Convert to dicts
        result_dicts = [result.to_dict() for result in results]
        
        # Close session
        session.close()
        
        # Convert to DataFrame
        if result_dicts:
            return pd.DataFrame(result_dicts)
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error getting user history: {e}")
        return pd.DataFrame()

def get_substance_statistics():
    """
    Get statistics about substance predictions from the database
    
    Returns:
    - stats: DataFrame with substance statistics
    """
    try:
        # Create SQL query
        query = """
        SELECT 
            substance,
            COUNT(*) as total_predictions,
            SUM(CASE WHEN prediction = true THEN 1 ELSE 0 END) as predicted_usage,
            AVG(probability) as average_probability
        FROM prediction_results
        GROUP BY substance
        ORDER BY total_predictions DESC
        """
        
        # Execute query
        engine = get_database_connection()
        stats = pd.read_sql(query, engine)
        
        return stats
    except Exception as e:
        st.error(f"Error getting substance statistics: {e}")
        return pd.DataFrame()

def get_risk_level_distribution():
    """
    Get distribution of risk levels from the database
    
    Returns:
    - stats: DataFrame with risk level distribution
    """
    try:
        # Create SQL query
        query = """
        SELECT 
            risk_level,
            COUNT(*) as count
        FROM prediction_results
        WHERE risk_level IS NOT NULL
        GROUP BY risk_level
        ORDER BY count DESC
        """
        
        # Execute query
        engine = get_database_connection()
        stats = pd.read_sql(query, engine)
        
        return stats
    except Exception as e:
        st.error(f"Error getting risk level distribution: {e}")
        return pd.DataFrame()