 # model/train_fraud_model.py
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
import joblib


def main():
    print("Starting training script...")

    # 1. Load data
    print("Loading data...")
    df = pd.read_csv("data/creditcard.csv")

    # 2. Prepare data
    print("Preparing data...")
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # 3. Handle class imbalance with SMOTE
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print("After SMOTE, class counts:", np.bincount(y_res))

    # 4. Train / test split
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_res,
        y_res,
        test_size=0.2,
        random_state=42,
        stratify=y_res,
    )

    # 5. Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Train model
    print("Training model...")
    clf = RandomForestClassifier(
        n_estimators=10,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train_scaled, y_train)

    # 7. Evaluate
    print("Evaluating model...")
    y_pred = clf.predict(X_test_scaled)
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    # 8. Save model and scaler
    print("Saving model and scaler...")
    joblib.dump(clf, "model/fraud_model.pkl")
    joblib.dump(scaler, "model/fraud_scaler.pkl")
    print("Saved fraud_model.pkl and fraud_scaler.pkl")
    print("Training script finished.")


if __name__ == "__main__":
    main()