# import necessary libraries
import numpy as np
from sklearn.metrics import accuracy_score

# function to perform weighted ensemble using rf and lstm model predictions
def ensemble_predictions(rf_models, lstm_models, test_data, rf_threshold=0.7, lstm_return_threshold=0.02):
    # initialise ensemble results
    ensemble_results = {}

    # loop through ticker in rf models
    for ticker in rf_models.keys():
        # ensure ticker exists in lstm models
        if ticker not in lstm_models:
            # skip ticker
            continue

        # get rf probabilities
        rf_proba = rf_models[ticker]['probabilities']

        # get lstm predictions
        lstm_pred_scaled = lstm_models[ticker]['predictions']
        target_scaler = lstm_models[ticker]['target_scaler']

        # convert lstm predictions back to actual price
        lstm_pred_prices = target_scaler.inverse_transform(
            lstm_pred_scaled.reshape(-1, 1)
        ).flatten()

        test_prices = test_data[ticker]['Close'].values

        min_len = min(len(rf_proba), len(lstm_pred_prices))
        rf_proba = rf_proba[-min_len:]
        lstm_pred_prices = lstm_pred_prices[-min_len:]
        current_prices = test_prices[-min_len-1:-1]
        actual_future_prices = test_prices[-min_len:]

        # generate ensemble signals
        signals = []
        for i in range(min_len):
            signal = ensemble_signal(
                rf_proba[i],
                lstm_pred_prices[i],
                current_prices[i],
                rf_threshold,
                lstm_return_threshold
            )
            signals.append(signal)

        signals = np.array(signals)

        # calculate actual returns when signals are generated
        actual_returns = (actual_future_prices - current_prices) / current_prices

        # calculate performance metrics
        buy_signals = signals == 1
        if buy_signals.sum() > 0:
            # average return when model said to buy
            avg_return_on_signal = actual_returns[buy_signals].mean()
            win_rate = (actual_returns[buy_signals] > 0).mean() 
            signal_count = buy_signals.sum()
        else:
            avg_return_on_signal = 0
            win_rate = 0
            signal_count = 0

        ensemble_results[ticker] = {
            'signals': signals,
            'rf_probabilities': rf_proba,
            'lstm_predictions': lstm_pred_prices,
            'current_prices': current_prices,
            'actual_prices': actual_future_prices,
            'signal_count': signal_count,
            'avg_return_on_signal': avg_return_on_signal,
            'win_rate': win_rate
        }

        print(f"{ticker} - Signals: {signal_count}/{min_len} | "
              f"Avg Return: {avg_return_on_signal:.2%} | Win Rate: {win_rate:.2%}")
        
    if ensemble_results:
        total_signals = sum([r['signal_count'] for r in ensemble_results.values()])
        avg_return = np.mean([r['avg_return_on_signal'] for r in ensemble_results.values()])
        avg_win_rate = np.mean([r['win_rate'] for r in ensemble_results.values()])

        print(f"\n{'='*60}")
        print(f"Overall Ensemble Performance:")
        print(f"Total Signals Generated: {total_signals}")
        print(f"Average Return on Signals: {avg_return:.2%}")
        print(f"Average Win Rate: {avg_win_rate:.2%}")
        print(f"{'='*60}")

    return ensemble_results


def ensemble_signal(rf_proba, lstm_predicted_price, current_price, rf_threshold=0.7, lstm_threshold=0.02):
    lstm_return = (lstm_predicted_price - current_price) / current_price

    if rf_proba > rf_threshold and lstm_return > lstm_threshold:
        return 1
    else :
        return 0
