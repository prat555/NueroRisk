import os
import sys
import logging
from model_trainer import train_all_models
from database import initialize_database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('model_initialization.log')
    ]
)
logger = logging.getLogger(__name__)

def initialize_application():
    """
    Initialize the application by:
    1. Creating database tables
    2. Training and saving all models
    
    Returns:
    - success: Boolean indicating if initialization was successful
    """
    logger.info("Starting application initialization...")
    
    try:
        # Step 1: Initialize database
        logger.info("Initializing database...")
        db_success = initialize_database()
        
        if not db_success:
            logger.error("Database initialization failed")
            return False
        
        logger.info("Database initialized successfully")
        
        # Step 2: Train and save models
        logger.info("Training and saving models...")
        training_results = train_all_models(priority_only=False)
        
        if not training_results:
            logger.error("Model training failed")
            return False
        
        num_models = len(training_results)
        logger.info(f"Successfully trained and saved {num_models} models")
        
        # Log summary of trained models
        for drug, result in training_results.items():
            if result['metrics']:
                logger.info(f"Model for {drug}: Accuracy={result['metrics']['accuracy']:.4f}, F1={result['metrics']['f1']:.4f}")
        
        logger.info("Application initialization completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error during application initialization: {e}")
        return False

if __name__ == "__main__":
    # This script is meant to be run directly
    success = initialize_application()
    
    if success:
        print("Application initialization completed successfully.")
        sys.exit(0)
    else:
        print("Application initialization failed. Check logs for details.")
        sys.exit(1)