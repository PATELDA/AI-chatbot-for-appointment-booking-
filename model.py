from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(96, return_sequences=True, input_shape=input_shape, kernel_regularizer='l2'),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(48, kernel_regularizer='l2'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    return model

def build_random_forest():
    """Random Forest with GridSearchCV setup"""
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    return GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid=param_grid,
        cv=TimeSeriesSplit(n_splits=3),
        scoring='neg_mean_squared_error'
    )
