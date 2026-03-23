# import necessary libraries
import tensorflow as tf
import numpy as np
import time
import gc
import joblib
import json
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error

# function to train lstm model
def train_lstm(X_train_seq, y_train_seq, X_test_seq, y_test_seq, input_shape, epochs=50, batch_size=32):
    # define stacked lstm architecture
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    # compile the model with adam optimiser and binary crosssentropy loss
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    # early stopping to prevent overfitting
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # train the model
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0
    )

    # predict actual price
    y_pred = model.predict(X_test_seq, verbose=0).flatten()
    mae = mean_absolute_error(y_test_seq, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_seq, y_pred))

    return model, mae, rmse, y_pred

# function to train lstm models for all tickers
def train_all_lstm(training_data):
    # initialise lstm models
    lstm_models = {}
    metrics_summary = {}

    # loop through training data
    for ticker, data in training_data.items():
        # define input shape dynamically
        input_shape = (data['X_train_seq'].shape[1], data['X_train_seq'].shape[2])

        # train lstm model for current ticker
        model, mae, rmse, y_pred = train_lstm(
            data['X_train_seq'],
            data['y_train_seq'],
            data['X_test_seq'],
            data['y_test_seq'],
            input_shape,
            epochs=50,
            batch_size=32
        )

        # saved trained model
        model.save(f'saved_models/lstm_{ticker}.h5')

        # save scalers
        joblib.dump(data['feature_scaler'], f'saved_models/lstm_feature_scaler_{ticker}.pkl')
        joblib.dump(data['target_scaler'], f'saved_models/lstm_target_scaler_{ticker}.pkl')

        # store model and evaluation results
        lstm_models[ticker] = {
            'model': model,
            'mae': mae,
            'rmse': rmse,
            'predictions': y_pred,
            'feature_scaler': data['feature_scaler'],
            'target_scaler': data['target_scaler']
        }

        metrics_summary[ticker] = {
            'mae': float(mae),
            'rmse': float(rmse),
            'test_samples': len(data['y_test_seq'])
        }

        # print evaluation metrics
        print(f"{ticker} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        # clear model from memory
        del model
        tf.keras.backend.clear_session()
        gc.collect()
        # brief delay 
        time.sleep(1)

    # save metrics summary to json file
    with open('model_metrics/lstm_metrics.json', 'w') as f:  # ← Add
        json.dump(metrics_summary, f, indent=2)
    
    return lstm_models