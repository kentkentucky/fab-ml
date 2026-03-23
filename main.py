import pandas as pd

from data.feature_engineering import merge_data
from data.preprocess import prepare_all_rf_datasets, prepare_all_lstm_datasets
from models.rf_model import train_all_rf
from models.lstm_model import train_all_lstm
from models.ensemble import ensemble_predictions

news_sentiment_file_path = "datasets/news_sentiment.csv"
rf_data_file_path = "datasets/rf_market_data.csv"
lstm_data_file_path = "datasets/lstm_market_data.csv"
news_sentiment = pd.read_csv(news_sentiment_file_path)
rf_stock_data = pd.read_csv(rf_data_file_path)
lstm_stock_data = pd.read_csv(lstm_data_file_path)

rf_merged_data = merge_data(rf_stock_data, news_sentiment, model="rf")
lstm_merged_data = merge_data(lstm_stock_data, news_sentiment, model="lstm")

rf_training_data = prepare_all_rf_datasets(rf_merged_data)
lstm_training_data = prepare_all_lstm_datasets(lstm_merged_data)

rf_models = train_all_rf(rf_training_data)
lstm_models = train_all_lstm(lstm_training_data)

ensemble_results = ensemble_predictions(rf_models, lstm_models, test_data=lstm_merged_data, rf_threshold=0.7, lstm_return_threshold=0.02)
