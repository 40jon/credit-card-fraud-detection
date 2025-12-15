import pandas as pd
import joblib

# 1. Load trained model and scaler
model = joblib.load("model/fraud_model.pkl")
scaler = joblib.load("model/fraud_scaler.pkl")

# 2. Load some data to test on
df = pd.read_csv("data/creditcard.csv")
X = df.drop("Class", axis=1)

# 3. Take one real transaction and predict
sample = X.iloc[0:1]
sample_scaled = scaler.transform(sample)
pred = model.predict(sample_scaled)

print("Predicted class for first row:", pred[0])