# import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import gc
import json

# function to train rf model
def train_rf(X_train, y_train, X_test, y_test):
    # initialise rf classifier with predefined hyperparameters
    rf_model = RandomForestClassifier(
        n_estimators=100,
        criterion='gini',
        max_depth=None,
        min_samples_split=3,
        random_state=42,
        n_jobs=1,
        verbose=0
    )

    # train the model on training dataset
    rf_model.fit(X_train, y_train)

    # generate predicted class for the test set
    y_pred = rf_model.predict(X_test)
    # generate predicted probabilities for the positive class
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

    # compute classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    return rf_model, accuracy, precision, recall, f1, y_pred_proba

def train_all_rf(training_data):
    # initialise rf models and metrics summary
    rf_models = {}
    metrics_summary = {}

    # loop through training data
    for ticker, data in training_data.items():
        # train model for current ticker
        model, acc, precision, recall, f1, proba = train_rf(
            data['X_train'],
            data['y_train'],
            data['X_test'],
            data['y_test']
        )

        # save model into folder
        joblib.dump(model, f'saved_models/rf_{ticker}.pkl')
        joblib.dump(data['scaler'], f'saved_models/rf_scaler_{ticker}.pkl')

        # store model and evaluation results
        rf_models[ticker] = {
            'model': model,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'probabilities': proba,
            'scaler': data['scaler']
        }

        # store metrics summary
        metrics_summary[ticker] = {
            'accuracy': float(acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'test_samples': len(data['y_test'])
        }


        # print all metrics
        print(f"{ticker} - Acc: {acc*100:.2f}% | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")

        # delete model reference
        del model
        gc.collect()

    # save metrics summary to json file
    with open('model_metrics/rf_metrics.json', 'w') as f:
        json.dump(metrics_summary, f, indent=2)

    return rf_models