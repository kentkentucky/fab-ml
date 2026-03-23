# import necessary libraries
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# function to prepare rf dataset for model training
def prepare_rf_dataset(data, test_size=0.2):
    # define feature columns
    feature_columns = [
        'rsi', 'sma_50', 'adx', 'volume', 'corr', 'prev_open_close', 'prev_close_high',
        'prev_close_low', 'momentum', 'volatility', 'sentiment_mean', 'sentiment_std',
        'news_count', 'sentiment_strength', 'sentiment_volume'
    ]

    # drop na values in defined feature columns and target
    data_clean = data.dropna(subset=feature_columns)

    # split dataset into x and y
    X = data_clean[feature_columns]
    y = data_clean['signal']

    # get split index
    split_idx = int(len(X) * (1 - test_size))

    # get training dataset
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    # scale training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # return categorise data
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train.values,
        'y_test': y_test.values,
        'scaler': scaler,
        'feature_columns': feature_columns
    }

# function to prepare all rf data
def prepare_all_rf_datasets(final_data):
    # initialise training data
    training_data = {}

    # loop through final data
    for ticker, data in final_data.items():
        try:
            # prepare and clean dataset
            prepared = prepare_rf_dataset(data)
            # save each data according to ticker
            training_data[ticker] = prepared

        except Exception as e:
            # error message
            print(f"Error: {e}")

    # return training data
    return training_data

# function to prepare and clean lstm dataset for model training
def prepare_lstm_dataset(data, test_size=0.2):
    # define feature columns
    feature_columns = [
        'rsi', 'sma_50', 'adx', 'volume', 'corr', 'momentum', 
        'volatility', 'macd', 'returns', 'sentiment_mean', 'sentiment_std',
        'news_count', 'sentiment_strength', 'sentiment_volume'
    ]

    # drop na values in defined feature columns and target
    data_clean = data.dropna(subset=feature_columns + ['target'])

    # split dataset into x and y
    X = data_clean[feature_columns]
    y = data_clean['target']

    # get index of split
    split_idx = int(len(X) * (1 - test_size))

    # get training dataset
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    # scale data between 0 - 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # scale target separately
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

    # create 60 day sequence for lstm model
    X_train_seq, y_train_seq = lstm_sequence(X_train_scaled, y_train_scaled)
    X_test_seq, y_test_seq = lstm_sequence(X_test_scaled, y_test_scaled)

    # return categorise training data
    return {
        'X_train_seq': X_train_seq,
        'X_test_seq': X_test_seq,
        'y_train_seq': y_train_seq,
        'y_test_seq': y_test_seq,
        'feature_scaler': scaler,
        'target_scaler': target_scaler,
        'feature_columns': feature_columns
    }

# function to create lstm sequence
def lstm_sequence(data, labels, length=60):
    # initialise x and y seq
    X_seq, y_seq = [], []

    # ensure consistent indexing
    labels_array = labels.values if hasattr(labels, 'values') else np.array(labels)

    # create sequence
    for i in range(length, len(data)):
        # append sequence into x seq
        X_seq.append(data[i - length: i])
        # append sequence into y seq
        y_seq.append(labels_array[i])

    # return sequence array
    return np.array(X_seq), np.array(y_seq)

# function to prepare all lstm dataset
def prepare_all_lstm_datasets(final_data):
    # initialise training data
    training_data = {}

    # loop through final data
    for ticker, data in final_data.items():
        try:
            # prepare and clean dataset
            prepared = prepare_lstm_dataset(data)
            # label data according to ticker
            training_data[ticker] = prepared

        except Exception as e:
            # print error message
            print(f"Error: {e}")

    # return training data
    return training_data