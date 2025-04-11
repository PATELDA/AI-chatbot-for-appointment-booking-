import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import joblib, os, logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPAppointmentPredictor:
    def __init__(self):
        self.scalers = {}
        self.label_encoders = {}
        self.model = None
        self.sequence_length = 7
        # Updated column names to match your CSV
        self.categorical_columns = [
            'GP_CODE', 'GP_NAME', 'SUPPLIER', 'PCN_CODE', 'PCN_NAME',
            'SUB_ICB_LOCATION_CODE', 'SUB_ICB_LOCATION_NAME', 'HCP_TYPE',
            'APPT_MODE', 'NATIONAL_CATEGORY', 'TIME_BETWEEN_BOOK_AND_APPT',
            'APPT_STATUS'
        ]

    def load_data(self, filepath):
        try:
            # df = pd.read_csv(filepath)
            files = [f for f in os.listdir(filepath) if f.endswith('.csv')]
            dfs = [pd.read_csv(os.path.join(filepath, f)) for f in files]
            df = pd.concat(dfs)
            df['APPOINTMENT_MONTH_START_DATE'] = pd.to_datetime(df['APPOINTMENT_MONTH_START_DATE'], format='%d%b%Y')
            logger.info("Data loaded successfully")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def create_advanced_features(self, df):
        try:
            # Time-based features
            df['year'] = df['APPOINTMENT_MONTH_START_DATE'].dt.year
            df['month'] = df['APPOINTMENT_MONTH_START_DATE'].dt.month
            df['day'] = df['APPOINTMENT_MONTH_START_DATE'].dt.day
            df['day_of_week'] = df['APPOINTMENT_MONTH_START_DATE'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['is_month_start'] = df['APPOINTMENT_MONTH_START_DATE'].dt.is_month_start.astype(int)
            df['is_month_end'] = df['APPOINTMENT_MONTH_START_DATE'].dt.is_month_end.astype(int)
            
            # Appointment type features
            df['is_face_to_face'] = (df['APPT_MODE'] == 'Face-to-Face').astype(int)
            df['is_telephone'] = (df['APPT_MODE'] == 'Telephone').astype(int)
            df['is_video'] = df['APPT_MODE'].str.contains('Video', na=False).astype(int)
            
            # Status features
            df['is_attended'] = (df['APPT_STATUS'] == 'Attended').astype(int)
            df['is_dna'] = (df['APPT_STATUS'] == 'DNA').astype(int)
            
            # Time between booking features
            df['is_same_day'] = (df['TIME_BETWEEN_BOOK_AND_APPT'] == 'Same Day').astype(int)
            df['is_within_week'] = df['TIME_BETWEEN_BOOK_AND_APPT'].isin(['Same Day', '1 Day', '2 to 7 Days']).astype(int)
            
            # HCP type features
            df['is_gp'] = (df['HCP_TYPE'] == 'GP').astype(int)
            
            # Aggregated features by GP_CODE
            df['avg_appointments_per_gp'] = df.groupby('GP_CODE')['COUNT_OF_APPOINTMENTS'].transform('mean')
            df['max_appointments_per_gp'] = df.groupby('GP_CODE')['COUNT_OF_APPOINTMENTS'].transform('max')
            
            # Rolling statistics
            df['appointments_rolling_mean_7d'] = df.groupby('GP_CODE')['COUNT_OF_APPOINTMENTS'].transform(
                lambda x: x.rolling(7, min_periods=1).mean())
            df['appointments_rolling_std_7d'] = df.groupby('GP_CODE')['COUNT_OF_APPOINTMENTS'].transform(
                lambda x: x.rolling(7, min_periods=1).std())
            
            # Attendance rates
            df['attendance_rate'] = df.groupby('GP_CODE')['is_attended'].transform('mean')
            df['dna_rate'] = df.groupby('GP_CODE')['is_dna'].transform('mean')
            df['face_to_face_ratio'] = df.groupby('GP_CODE')['is_face_to_face'].transform('mean')
            
            # Encode categorical variables
            for col in self.categorical_columns:
                self.label_encoders[col] = LabelEncoder()
                df[col + '_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
            
            # Handle missing values using modern syntax
            df = df.ffill().bfill()
            
            logger.info("Feature engineering completed successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            raise
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            raise

    def prepare_sequences(self, data, sequence_length):
        """
        Modified to handle the data without grouping by GP_CODE
        """
        X, y = [], []
        data_values = data.values
        for i in range(len(data_values) - sequence_length):
            X.append(data_values[i:(i + sequence_length)])
            y.append(data_values[i + sequence_length][-1])  # COUNT_OF_APPOINTMENTS is the target
        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        model = Sequential([
            Bidirectional(LSTM(256, return_sequences=True, activation='relu'), 
                         input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.3),
            
            Bidirectional(LSTM(128, return_sequences=True)),
            BatchNormalization(),
            Dropout(0.3),
            
            Bidirectional(LSTM(64)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        return model

    def train_model(self, X, y, validation_split=0.2):
        try:
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=7,
                    min_lr=0.00001,
                    verbose=1
                ),
                ModelCheckpoint(
                    'best_model.keras',
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            ]

            history = self.model.fit(
                X, y,
                epochs=20,
                batch_size=32,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )
            
            # Plot training history
            self.plot_training_history(history)
            
            return history
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise

    def plot_training_history(self, history):
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()

    def evaluate_model(self, X_test, y_test):
        try:
            predictions = self.model.predict(X_test)
            
            metrics = {
                'MSE': mean_squared_error(y_test, predictions),
                'RMSE': np.sqrt(mean_squared_error(y_test, predictions)),
                'MAE': mean_absolute_error(y_test, predictions),
                'R2': r2_score(y_test, predictions)
            }
            
            # Plot predictions
            plt.figure(figsize=(15, 6))
            plt.plot(y_test[:100], label='Actual', marker='o')
            plt.plot(predictions[:100], label='Predicted', marker='x')
            plt.title('Actual vs Predicted Appointments (First 100 samples)')
            plt.xlabel('Sample Index')
            plt.ylabel('Number of Appointments')
            plt.legend()
            plt.savefig('prediction_results.png')
            plt.close()
            
            return metrics, predictions
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            raise

def main():
    predictor = GPAppointmentPredictor()
    
    # Load and prepare data
    df = predictor.load_data('Practice_Level_Crosstab_Jan_24')
    df = predictor.create_advanced_features(df)

    # Select features for model (updated to match your CSV columns)
    feature_columns = [
        'COUNT_OF_APPOINTMENTS', 'year', 'month', 'day', 'day_of_week',
        'is_weekend', 'is_month_start', 'is_month_end', 'is_face_to_face',
        'is_telephone', 'is_video', 'is_attended', 'is_dna', 'is_same_day',
        'is_within_week', 'is_gp', 'avg_appointments_per_gp',
        'max_appointments_per_gp', 'appointments_rolling_mean_7d',
        'appointments_rolling_std_7d', 'attendance_rate', 'dna_rate',
        'face_to_face_ratio'
    ]
    
    # Add encoded categorical columns
    for col in predictor.categorical_columns:
        if col + '_encoded' in df.columns:
            feature_columns.append(col + '_encoded')

    # Prepare sequences
    X, y = predictor.prepare_sequences(df[feature_columns], predictor.sequence_length)

    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build and train model
    predictor.model = predictor.build_model((predictor.sequence_length, len(feature_columns)))
    history = predictor.train_model(X_train, y_train)

    # Evaluate model
    metrics, predictions = predictor.evaluate_model(X_test, y_test)
    logger.info("Model Metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value}")

    # Save model and preprocessing objects
    predictor.model.save('gp_appointment_prediction_model.keras')
    joblib.dump(predictor.label_encoders, 'label_encoders.pkl')

if __name__ == "__main__":
    main()