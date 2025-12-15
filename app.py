from flask import Flask, request, jsonify
import joblib
import numpy as np
import datetime
import pandas as pd  # used to build a row matching the model


app = Flask(__name__)


# Load model and scaler at startup
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("fraud_scaler.pkl")


def build_feature_vector(amount, time_value, extra_features=None):
    """
    Build a single-row DataFrame to match the training columns.
    Assumes creditcard.csv has 'Time', 'Amount', and V1..V28 columns.
    For now, use Amount and Time; set V1..V28 to 0 unless provided.
    """
    if extra_features is None:
        extra_features = {}

    data = {
        "Time": time_value,
        "Amount": amount,
    }

    # Add V1..V28 as zeros if not provided
    for i in range(1, 29):
        col = f"V{i}"
        data[col] = extra_features.get(col, 0.0)

    df = pd.DataFrame([data])
    return df


def generate_notifications(transaction):
    """
    Build the user-facing texts (SMS, email, phone, app push, etc.).
    """
    amount = transaction.get("amount", 0.0)
    merchant = transaction.get("merchant", "UNKNOWN MERCHANT")
    date_str = transaction.get("date", datetime.date.today().isoformat())
    bank_name = transaction.get("bank_name", "Your Bank")

    sms = (
        f"{bank_name} Fraud Alert: Did you authorize a charge of "
        f"${amount:.2f} at {merchant} on {date_str}? "
        "Reply YES or NO. Do not include personal info."
    )

    email_subject = "Suspicious activity on your account"
    email_body = (
        f"We detected a suspicious charge of ${amount:.2f} at {merchant} on {date_str}. "
        f"Please log in to your {bank_name} account directly (not by clicking links in this email) "
        "to review and confirm whether this was you."
    )

    phone_script = (
        f"Hello, this is the fraud department at {bank_name}. "
        f"We noticed a charge of ${amount:.2f} at {merchant} on {date_str}. "
        "We are calling to verify whether you recognize this transaction."
    )

    mobile_push = "Suspicious transaction detected. Tap to review."

    return {
        "sms": sms,
        "email_subject": email_subject,
        "email_body": email_body,
        "phone_call_script": phone_script,
        "mobile_push": mobile_push,
    }


@app.route("/analyze_transaction", methods=["POST"])
def analyze_transaction():
    """
    Example request JSON body:

    {
        "amount": 123.45,
        "time": 100000,
        "merchant": "AMAZON",
        "date": "2025-11-25",
        "bank_name": "Sample Bank"
    }
    """
    data = request.get_json(silent=True) or {}

    # Basic validation
    if "amount" not in data or "time" not in data:
        return jsonify({"error": "Fields 'amount' and 'time' are required"}), 400

    amount = float(data.get("amount", 0.0))
    time_value = float(data.get("time", 0.0))

    # 1. Build feature vector
    feature_df = build_feature_vector(amount, time_value)

    # 2. Scale features (same scaler as training)
    feature_scaled = scaler.transform(feature_df)

    # 3. Predict fraud
    fraud_prob = model.predict_proba(feature_scaled)[0, 1]
    prediction = model.predict(feature_scaled)[0]
    is_fraud = bool(prediction == 1)

    # 4. Build alerts / UI messages
    notifications = generate_notifications(data)

    if is_fraud:
        transaction_status = "pending"
        banner = (
            "Your account is temporarily locked due to unusual activity. "
            "Please contact the fraud department."
        )
        question = "Do you recognize this transaction? Yes / No"
    else:
        transaction_status = "posted"
        banner = ""
        question = ""

    response = {
        "is_fraud": is_fraud,
        "fraud_probability": float(fraud_prob),
        "notifications": notifications,
        "online_banking_view": {
            "transaction_status": transaction_status,
            "banner": banner,
            "question": question,
            "transaction": {
                "amount": amount,
                "merchant": data.get("merchant", "UNKNOWN MERCHANT"),
                "date": data.get("date", datetime.date.today().isoformat()),
            },
        },
    }

    return jsonify(response)


@app.route("/", methods=["GET"])
def home():
    return (
        "Credit Card Fraud Detection API is running. "
        "POST JSON to /analyze_transaction to test."
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)