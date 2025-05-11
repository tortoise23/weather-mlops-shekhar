import pandas as pd
import joblib
import json
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def evaluate_model():
    # Load model and test data
    model = joblib.load("models/xgb_model.joblib")
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv")

    # Predict
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()  # convert to list for JSON

    results = {
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm
    }

    # Save to JSON
    with open("evaluation/metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Evaluation complete. Metrics saved to evaluation/metrics.json")

if __name__ == "__main__":
    evaluate_model()
