# ensure project root directory is accessible for module imports
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# import necessary libraries
import backtrader as bt
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras

from data.preprocess import lstm_sequence
from data.feature_engineering import merge_data

# ensemble strategy class for backtrader
class EnsembleStrategy(bt.Strategy):
    params = (
        # signals from ensemble model
        ('ensemble_signals', None),
        # base allocation per trade
        ('base_position_size', 0.1),
        # max allocation cap
        ('max_position_size', 0.3),
        # stop loss
        ('stop_loss_pct', 0.1),
        # take profit
        ('take_profit_pct', 0.2),
    )

    def __init__(self):
        # store ensemble signals for each ticker
        self.ensemble_signals = self.params.ensemble_signals
        self.order = None
        # track signal index for each ticker
        self.signal_idx = {}
        self.entry_prices = {}

        # create signal index for each ticker
        for i, d in enumerate(self.datas):
            ticker = d._name
            self.signal_idx[ticker] = 0

    def next(self):
        for i, d in enumerate(self.datas):
            ticker = d._name
            position = self.getposition(d)
            skip_signal = False

            if position:
                current_price = d.close[0]
                entry_price = self.entry_prices.get(ticker, current_price)
                pnl_pct = (current_price - entry_price) / entry_price
                
                # stop loss
                if pnl_pct <= -self.params.stop_loss_pct:
                    self.close(data=d)
                    print(f'{d.datetime.date(0)} STOP LOSS {ticker}: {pnl_pct*100:.2f}%')
                    if ticker in self.entry_prices:
                        del self.entry_prices[ticker]
                    skip_signal = True
                
                # take profit
                if pnl_pct >= self.params.take_profit_pct:
                    self.close(data=d)
                    print(f'{d.datetime.date(0)} TAKE PROFIT {ticker}: {pnl_pct*100:.2f}%')
                    if ticker in self.entry_prices:
                        del self.entry_prices[ticker]
                    skip_signal = True

            # check if we have signals for this ticker
            if ticker not in self.ensemble_signals:
                continue

            signals = self.ensemble_signals[ticker]['signals']
            idx = self.signal_idx[ticker]

            # skip if signals exhausted
            if idx >= len(signals):
                continue
            
            signal = signals[idx]

            if not skip_signal:
                # buy signal 
                if signal == 1 and not position:
                    # get signal strength / confidence
                    rf_proba = self.ensemble_signals[ticker]['rf_probabilities'][idx]
                    lstm_pred = self.ensemble_signals[ticker]['lstm_predictions'][idx]
                    current_price_signal = self.ensemble_signals[ticker]['current_prices'][idx]
                    predicted_return = (lstm_pred - current_price_signal) / current_price_signal

                    # scale position by confidence
                    confidence_multiplier = min(rf_proba / 0.65, 2.0)
                    return_multiplier = min(max((predicted_return - 0.04) / 0.06, 1.0), 2.0)

                    position_size = self.params.base_position_size * confidence_multiplier * return_multiplier
                    position_size = min(position_size, self.params.max_position_size)

                    size = int(self.broker.getvalue() * position_size / d.close[0])
                    if size > 0:
                        self.buy(data=d, size=size)
                        self.entry_prices[ticker] = d.close[0]
                        print(f'{d.datetime.date()} BUY {ticker}: {size} shares @ ${d.close[0]:.2f}')

                # sell signal
                elif signal == 0 and position:
                    self.close(data=d)
                    print(f'{d.datetime.date()} SELL {ticker}: {position.size} shares @ ${d.close[0]:.2f}')
                    if ticker in self.entry_prices:
                            del self.entry_prices[ticker]
            
            # move to next signal index
            self.signal_idx[ticker] += 1

# load trained models and scalers
def load_models(tickers):
    rf_models = {}
    lstm_models = {}

    for ticker in tickers:
        try:
            # load rf model and scaler
            rf_model = joblib.load(f'saved_models/rf_{ticker}.pkl')
            rf_scaler = joblib.load(f'saved_models/rf_scaler_{ticker}.pkl')

            rf_models[ticker] = {
                'model': rf_model,
                'scaler': rf_scaler
            }

            # load lstm model and scalers
            lstm_model = keras.models.load_model(
                f'saved_models/lstm_{ticker}.h5',
                custom_objects={'mse': keras.losses.MeanSquaredError()}
            )
            feature_scaler = joblib.load(f'saved_models/lstm_feature_scaler_{ticker}.pkl')
            target_scaler = joblib.load(f'saved_models/lstm_target_scaler_{ticker}.pkl')

            lstm_models[ticker] = {
                'model': lstm_model,
                'feature_scaler': feature_scaler,
                'target_scaler': target_scaler,
            }

            print(f"Loaded models for {ticker}")

        except Exception as e:
            print(f"Error loading models for {ticker}: {e}")

    return rf_models, lstm_models

# generate predictions using the loaded models
def generate_predictions(rf_models, lstm_models, backtest_data, rf_threshold=0.7, lstm_threshold=0.1):
    predictions = {}

    for ticker in rf_models.keys():
        if ticker not in lstm_models or ticker not in backtest_data:
            continue

        print(f'Generating predictions for {ticker}...')

        try:
            data = backtest_data[ticker].copy()

            # rf feature columns
            rf_feature_columns = [
                'rsi', 'sma_50', 'adx', 'volume', 'corr', 'prev_open_close', 'prev_close_high',
                'prev_close_low', 'momentum', 'volatility', 'sentiment_mean', 'sentiment_std',
                'news_count', 'sentiment_strength', 'sentiment_volume'
            ]

            # lstm feature columns
            lstm_feature_columns = [
                'rsi', 'sma_50', 'adx', 'volume', 'corr', 'momentum', 
                'volatility', 'macd', 'returns', 'sentiment_mean', 'sentiment_std',
                'news_count', 'sentiment_strength', 'sentiment_volume'
            ]

            # Check if all required features exist
            missing_rf = [col for col in rf_feature_columns if col not in data.columns]
            missing_lstm = [col for col in lstm_feature_columns if col not in data.columns]
            
            if missing_rf:
                print(f"Missing RF features for {ticker}: {missing_rf}")
                continue
            if missing_lstm:
                print(f"Missing LSTM features for {ticker}: {missing_lstm}")
                continue

            # prepare rf features
            X_rf = data[rf_feature_columns].dropna()
            X_rf_scaled = rf_models[ticker]['scaler'].transform(X_rf)

            # get rf predictions
            rf_proba = rf_models[ticker]['model'].predict_proba(X_rf_scaled)[:, 1]

            # prepare lstm features
            X_lstm = data[lstm_feature_columns].dropna()
            X_lstm_scaled = lstm_models[ticker]['feature_scaler'].transform(X_lstm)

            # create lstm sequence
            X_lstm_seq, _ = lstm_sequence(X_lstm_scaled, np.zeros(len(X_lstm_scaled)), length=60)

            # get lstm predictions
            lstm_pred_scaled = lstm_models[ticker]['model'].predict(X_lstm_seq, verbose=0).flatten()
            lstm_pred_prices = lstm_models[ticker]['target_scaler'].inverse_transform(
                lstm_pred_scaled.reshape(-1, 1)
            ).flatten()

            # align lengths
            min_len = min(len(rf_proba), len(lstm_pred_prices))
            rf_proba = rf_proba[-min_len:]
            lstm_pred_prices = lstm_pred_prices[-min_len:]

            # get corresponding prices
            current_prices = data['Close'].values[-min_len-1:-1]
            future_prices = data['Close'].values[-min_len:]

            # generate ensemble signals
            signals = []
            for i in range(min_len):
                lstm_return = (lstm_pred_prices[i] - current_prices[i]) / current_prices[i]

                # both models must agree
                if rf_proba[i] > rf_threshold and lstm_return > lstm_threshold:
                    signals.append(1)
                else:
                    signals.append(0)

            predictions[ticker] = {
                'signals': np.array(signals),
                'rf_probabilities': rf_proba,
                'lstm_predictions': lstm_pred_prices,
                'current_prices': current_prices,
                'actual_prices': future_prices
            }

            signal_count = np.sum(signals)
            print(f"Generated {signal_count} BUY signals out of {len(signals)} periods")

        except Exception as e:
            print(f"Error generating predictions for {ticker}: {e}")

    return predictions

def backtest(ensemble_results, price_data, initial_capital=10000):
    if not ensemble_results:
        print("\nNo predictions available. Cannot run backtest.")
        return None
    
    cerebro = bt.Cerebro()

    for ticker, df in price_data.items():
        if ticker not in ensemble_results:
            continue
            
        # ensure dataframe has datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # create backtrader data feed
        data = bt.feeds.PandasData(
            dataname=df,
            name=ticker,
            datetime=None,
            open='Open',
            high='High',
            low='Low',
            close='Close',
            volume='Volume',
            openinterest=-1
        )
        cerebro.adddata(data)

    # add strategy 
    cerebro.addstrategy(EnsembleStrategy, ensemble_signals=ensemble_results)

    # set initial capital
    cerebro.broker.setcash(initial_capital)

    # set commission
    cerebro.broker.setcommission(commission=0.001)

    # add analysers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # run backtest
    print(f'\nStarting Portfolio Value: ${cerebro.broker.getvalue():,.2f}')
    results = cerebro.run()
    print(f'Final Portfolio Value: ${cerebro.broker.getvalue():,.2f}')

    # extract results
    strat = results[0]
    
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)

    sharpe_analysis = strat.analyzers.sharpe.get_analysis()
    sharpe = sharpe_analysis.get('sharperatio', None)
    if sharpe is not None:
        print(f"Sharpe Ratio: {sharpe:.2f}")
    else:
        print("Sharpe Ratio: N/A")
    
    print(f"Max Drawdown: {strat.analyzers.drawdown.get_analysis()['max']['drawdown']:.2f}%")
    print(f"Total Return: {strat.analyzers.returns.get_analysis()['rtot']*100:.2f}%")
    
    trade_analysis = strat.analyzers.trades.get_analysis()
    
    # use .get() for safe dictionary access
    total_closed = trade_analysis.get('total', {}).get('closed', 0)
    won = trade_analysis.get('won', {}).get('total', 0)
    lost = trade_analysis.get('lost', {}).get('total', 0)
    
    print(f"\nTotal Trades: {total_closed}")
    print(f"Won: {won}")
    print(f"Lost: {lost}")
    
    if total_closed > 0:
        win_rate = (won / total_closed) * 100
        print(f"Win Rate: {win_rate:.2f}%")
    
    print("="*60)
    
    return cerebro


# main execution
print("="*60)
print("LOADING DATA")
print("="*60)

tickers = ['AAPL', 'GOOGL', 'NVDA', 'MSFT', 'TSLA', 'JPM', 'V', 'JNJ', 'AMZN', 'WMT']
news_sentiment = pd.read_csv('./datasets/news_sentiment.csv')
rf_stock_data = pd.read_csv('./datasets/rf_market_data.csv')
lstm_stock_data = pd.read_csv('./datasets/lstm_market_data.csv')

print("Merging RF data...")
backtest_data_rf = merge_data(rf_stock_data, news_sentiment, model='rf')
print("Merging LSTM data...")
backtest_data_lstm = merge_data(lstm_stock_data, news_sentiment, model='lstm')

# combine both datasets to get all features
print("\nCombining features from both datasets...")
backtest_data = {}
for ticker in tickers:
    if ticker in backtest_data_rf and ticker in backtest_data_lstm:
        # start with lstm data
        combined = backtest_data_lstm[ticker].copy()
        
        # add missing rf specific features
        rf_specific = ['prev_open_close', 'prev_close_high', 'prev_close_low']
        for col in rf_specific:
            if col in backtest_data_rf[ticker].columns and col not in combined.columns:
                combined[col] = backtest_data_rf[ticker][col]

        # ensure datetime index exists
        if 'Date' in combined.columns:
            combined['Date'] = pd.to_datetime(combined['Date'])
            combined = combined.set_index('Date')
        elif not isinstance(combined.index, pd.DatetimeIndex):
            # if no date column and index isn't datetime, try to convert index
            combined.index = pd.to_datetime(combined.index)
        
        backtest_data[ticker] = combined
        print(f"Combined features for {ticker}: {len(combined.columns)} columns")

# load models
print("\n" + "="*60)
print("LOADING MODELS")
print("="*60)
rf_models, lstm_models = load_models(tickers)

# generate predictions
print("\n" + "="*60)
print("GENERATING PREDICTIONS")
print("="*60)
predictions = generate_predictions(
    rf_models, 
    lstm_models, 
    backtest_data, 
    rf_threshold=0.7,
    lstm_threshold=0.1
)

# backtest
if predictions:
    print(f"\n Generated predictions for {len(predictions)} tickers")
    
    print("\n" + "="*60)
    print("RUNNING BACKTEST")
    print("="*60)
    cerebro = backtest(predictions, backtest_data, initial_capital=100000)
    
    print("\n Backtest complete!")
else:
    print("\n No predictions generated. Check feature engineering.")