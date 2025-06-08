import os
import logging
import time
import sys
from initialize_models import initialize_application
from database import initialize_database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app_startup.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    """
    Main startup function that initializes the application:
    1. Checks if required directories exist
    2. Initializes the database
    3. Trains and saves prediction models if they don't exist
    """
    logger.info("Starting application setup...")
    
    # Step 1: Check for required directories
    required_dirs = ["pretrained_models"]
    for directory in required_dirs:
        if not os.path.exists(directory):
            logger.info(f"Creating directory: {directory}")
            os.makedirs(directory)
    
    # Step 2: Initialize the database
    logger.info("Initializing database...")
    db_success = initialize_database()
    
    if not db_success:
        logger.error("Database initialization failed. Continuing anyway...")
    else:
        logger.info("Database initialized successfully")
    
    # Step 3: Initialize application (train/load models)
    logger.info("Initializing application models...")
    app_success = initialize_application()
    
    if not app_success:
        logger.error("Application initialization failed")
        return False
    
    logger.info("Application setup completed successfully")
    return True

if __name__ == "__main__":
    start_time = time.time()
    success = main()
    end_time = time.time()
    
    duration = end_time - start_time
    
    if success:
        logger.info(f"Setup completed successfully in {duration:.2f} seconds")
        print(f"Setup completed successfully in {duration:.2f} seconds")
        sys.exit(0)
    else:
        logger.error(f"Setup failed after {duration:.2f} seconds")
        print(f"Setup failed after {duration:.2f} seconds")
        sys.exit(1)