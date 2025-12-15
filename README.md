# Credit Card Fraud Detection

A simple machine learning project that detects fraudulent credit card transactions using a RandomForest model and exposes predictions via a Flask API.

## Features

- Trains a RandomForest classifier on the popular credit card fraud dataset, using SMOTE to balance classes.
- Saves the trained model and scaler to disk with joblib.
- Provides a Flask API with a `/predict` endpoint to get fraud/not‑fraud predictions for new transactions.

## Project structure

- `fraud_detect.py` – trains the model and saves `model/fraud_model.pkl` and `model/fraud_scaler.pkl`.
- `api.py` – Flask API that loads the saved model and scaler and exposes the `/predict` endpoint.
- `test_model.py` – simple script that loads the saved model and runs a test prediction on the first row of the dataset.
- `data/creditcard.csv` – input dataset (credit card transactions).
- `model/` – folder containing the saved model and scaler files.
- `frontend/` – placeholder for a UI or client that can call the API.

## How to run

1. Create and activate a virtual environment (first time only):

## Quick start (for me)

1. Activate venv: `source venv/bin/activate`
2. Train (if needed): `python3 fraud_detect.py`
3. Run API: `python3 api.py`
4. Test: `curl ... /predict ...`
