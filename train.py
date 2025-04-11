import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model import build_lstm_model, build_random_forest
from data_processing import create_sliding_windows
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import joblib

def train_pipeline():
    # Configure paths
    current_dir = os.path.dirname(__file__)
    data_dir = os.path.join(current_dir, '../../data')
    
    # Load and prepare data
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    # Create sequence data
    X_seq, X_flat, y = create_sliding_windows(np.concatenate((X_train, X_test)))

    # LSTM Training
    lstm_model = build_lstm_model((X_seq.shape[1], X_seq.shape[2]))
    lstm_model.compile(optimizer='adam', 
                     loss=tf.keras.losses.Huber(),
                     metrics=['mae'])

    model_dir = os.path.join(current_dir, '../../models')
    os.makedirs(model_dir, exist_ok=True)

    # Preserve existing callbacks with updated filenames
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint(os.path.join(model_dir, 'best_lstm.keras'), save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5),
        tf.keras.callbacks.TensorBoard(log_dir='../logs')
    ]

    print("Training LSTM model...")
    lstm_history = lstm_model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        callbacks=callbacks
    )

    # Random Forest Training
    print("\nTraining Random Forest model...")
    rf_model = build_random_forest()
    rf_model.fit(X_flat[:len(X_train)], y[:len(X_train)])

    # Evaluate both models
    print("\nEvaluating models:")
    lstm_preds = lstm_model.predict(X_test)
    rf_preds = rf_model.predict(X_flat[len(X_train):])

    # Generate comparison plot
    plt.figure(figsize=(12,6))
    plt.plot(y_test, label='True Values')
    plt.plot(lstm_preds, label='LSTM Predictions')
    plt.plot(rf_preds, label='Random Forest Predictions')
    plt.title('Model Predictions Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Appointments')
    plt.legend()
    plt.savefig(os.path.join(model_dir, 'model_comparison.png'))
    plt.close()

    # Save models
    lstm_model.save(os.path.join(model_dir, 'lstm_model.keras'))
    joblib.dump(rf_model, os.path.join(model_dir, 'rf_model.pkl'))

    print(f"LSTM Test MAE: {mean_squared_error(y_test, lstm_preds, squared=False):.2f}")
    print(f"Random Forest Test MAE: {mean_squared_error(y_test, rf_preds, squared=False):.2f}")
    print(f"Best RF params: {rf_model.best_params_}")

if __name__ == "__main__":
    train_pipeline()
