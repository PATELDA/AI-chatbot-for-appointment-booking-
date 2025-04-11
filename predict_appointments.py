import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from datetime import datetime
from app import GPAppointmentPredictor  # Import your class from the training script

def load_model_and_encoders():
    model = tf.keras.models.load_model('gp_appointment_prediction_model.keras')
    label_encoders = joblib.load('label_encoders.pkl')
    return model, label_encoders

def prepare_input(df, predictor, label_encoders):
    # Manually encode the categorical features
    for col in predictor.categorical_columns:
        encoder = label_encoders.get(col)
        if encoder:
            df[col] = df[col].astype(str)
            df[col + '_encoded'] = df[col].apply(
                lambda x: encoder.transform([x])[0] if x in encoder.classes_ else 0
            )
        else:
            df[col + '_encoded'] = 0

    # Apply feature engineering
    df = predictor.create_advanced_features(df)

    # Feature selection
    feature_columns = [
        'COUNT_OF_APPOINTMENTS', 'year', 'month', 'day', 'day_of_week',
        'is_weekend', 'is_month_start', 'is_month_end', 'is_face_to_face',
        'is_telephone', 'is_video', 'is_attended', 'is_dna', 'is_same_day',
        'is_within_week', 'is_gp', 'avg_appointments_per_gp',
        'max_appointments_per_gp', 'appointments_rolling_mean_7d',
        'appointments_rolling_std_7d', 'attendance_rate', 'dna_rate',
        'face_to_face_ratio'
    ] + [col + '_encoded' for col in predictor.categorical_columns]

    df = df[feature_columns]
    return df

def predict_appointment(input_data: pd.DataFrame):
    predictor = GPAppointmentPredictor()
    model, label_encoders = load_model_and_encoders()

    predictor.model = model
    predictor.label_encoders = label_encoders

    df = prepare_input(input_data, predictor, label_encoders)

    # Create input sequence
    sequence = df.tail(predictor.sequence_length).values
    sequence = sequence.reshape((1, predictor.sequence_length, df.shape[1]))

    prediction = model.predict(sequence)
    return prediction[0][0]

# Example usage
if __name__ == "__main__":
    # Simulated latest 7-day input for a GP
    data = {
        'APPOINTMENT_MONTH_START_DATE': pd.date_range('2024-04-01', periods=7),
        'GP_CODE': ['GP001'] * 7,
        'GP_NAME': ['Dr. A'] * 7,
        'SUPPLIER': ['XYZ'] * 7,
        'PCN_CODE': ['PCN001'] * 7,
        'PCN_NAME': ['PCN A'] * 7,
        'SUB_ICB_LOCATION_CODE': ['ICB001'] * 7,
        'SUB_ICB_LOCATION_NAME': ['Location A'] * 7,
        'HCP_TYPE': ['GP'] * 7,
        'APPT_MODE': ['Face-to-Face'] * 7,
        'NATIONAL_CATEGORY': ['Routine'] * 7,
        'TIME_BETWEEN_BOOK_AND_APPT': ['Same Day'] * 7,
        'APPT_STATUS': ['Attended'] * 7,
        'COUNT_OF_APPOINTMENTS': [20, 22, 21, 19, 23, 24, 25]
    }

    df_input = pd.DataFrame(data)
    prediction = predict_appointment(df_input)
    print(f"\n📅 Predicted next appointment count: {round(prediction)}")
