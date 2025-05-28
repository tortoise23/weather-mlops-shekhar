import pandas as pd
import joblib

def predict_sample(index=5):
    # Load test data and labels
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv")
    model = joblib.load("models/xgb_model.joblib")

    # Get one sample and true label
    sample = X_test.iloc[[index]]
    true_label = y_test.iloc[index].item()

    # Make prediction
    pred = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0][1]

    # Display result
    print(f"Prediction for sample {index}:")
    print(f"RainTomorrow: {'Yes' if pred == 1 else 'No'} (probability: {prob:.2%})")
    print(f"Actual      : {'Yes' if true_label == 1 else 'No'}")
    print("Correct!" if pred == true_label else "Incorrect.")

if __name__ == "__main__":
    predict_sample()
