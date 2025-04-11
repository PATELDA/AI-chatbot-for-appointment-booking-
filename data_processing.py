import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import os
from datetime import datetime

def validate_feedback_data(df: pd.DataFrame) -> dict:
    """Basic validation for feedback data"""
    report = {
        'valid': True,
        'errors': [],
        'validated_at': datetime.now().isoformat()
    }
    
    # Required columns check
    required_cols = {'COUNT_OF_APPOINTMENTS', 'APPOINTMENT_MONTH_START_DATE'}
    missing = required_cols - set(df.columns)
    if missing:
        report['valid'] = False
        report['errors'].append(f"Missing columns: {missing}")
    
    # Date format validation
    try:
        pd.to_datetime(df['APPOINTMENT_MONTH_START_DATE'], format='%d%b%Y')
    except ValueError as e:
        report['valid'] = False 
        report['errors'].append(f"Date format error: {str(e)}")
    
    # Value range checks
    if (df['COUNT_OF_APPOINTMENTS'] < 0).any():
        report['valid'] = False
        report['errors'].append("Negative appointment counts found")
        
    return report

def save_versioned_dataset(data: pd.DataFrame, base_path: str = None) -> str:
    """Save dataset with timestamp versioning"""
    if base_path is None:
        base_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            "..", "..", "data", "versions"
        ))
    
    # Create directory structure if needed
    os.makedirs(base_path, exist_ok=True)
    
    version = datetime.now().strftime("%Y%m%d%H%M")
    filename = os.path.join(base_path, f"dataset_{version}.parquet")
    
    try:
        data.to_parquet(filename)
        print(f"Successfully saved versioned dataset to:\n{filename}")
        return filename
    except Exception as e:
        print(f"Failed to save dataset: {str(e)}")
        raise

def create_sliding_windows(data, window_size=14):
    """Create 3D sequences for LSTM and 2D arrays for Random Forest"""
    X_seq, y = [], []
    for i in range(len(data) - window_size):
        X_seq.append(data[i:(i + window_size)])
        y.append(data[i + window_size, 0])
    
    # Create 2D flattened version for tree-based models
    X_flat = np.array(X_seq).reshape(len(X_seq), -1)
    
    return np.array(X_seq), X_flat, np.array(y)

# Load and merge datasets with corrected path
current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(os.path.dirname(current_dir))
data_dir = os.path.join(project_root, 'Practice_Level_Crosstab_Jan_24')
files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
dfs = [pd.read_csv(os.path.join(data_dir, f)) for f in files]

# Merge and validate data
merged = pd.concat(dfs)
validation_report = validate_feedback_data(merged)
if not validation_report['valid']:
    raise ValueError(f"Data validation failed: {validation_report['errors']}")

# Save versioned dataset
versioned_path = save_versioned_dataset(merged)
print(f"Saved validated dataset version: {versioned_path}")

# Use verified column names from dataset
merged = merged.sort_values('APPOINTMENT_MONTH_START_DATE')

# Convert date with correct format
merged['Date'] = pd.to_datetime(merged['APPOINTMENT_MONTH_START_DATE'], format='%d%b%Y')

# Feature engineering
merged['day_of_week'] = merged['Date'].dt.dayofweek
merged['is_weekend'] = merged['day_of_week'].isin([5,6]).astype(int)

# Select features and target based on actual columns
features = ['COUNT_OF_APPOINTMENTS', 'day_of_week', 'is_weekend']
target = 'COUNT_OF_APPOINTMENTS'
data = merged[features].values

# Normalize
scaler = RobustScaler()
scaled_data = scaler.fit_transform(data)

# Create sequences
X, y = create_sliding_windows(scaled_data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Save processed data
output_dir = os.path.join(current_dir, '../../data')
os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
