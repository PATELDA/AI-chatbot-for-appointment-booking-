# config.py
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your_secret_key'
    MONGO_URI = 'mongodb://localhost:27017/db_booking'

    # Appointment settings
    APPOINTMENT_SLOTS = {
        'start_hour': 9,    # 9 AM
        'end_hour': 17,     # 5 PM
        'interval': 30      # 30 minutes
    }
    
    MAX_APPOINTMENTS_PER_DAY = 50
    
    # Data directory
    DATA_DIR = r"C:\Users\daksh\Downloads\Practice_Level_Crosstab_Jan_24 (1)"