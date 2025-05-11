import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import accuracy_score

def train_and_save_model():
    # Load preprocessed data
    X_train = pd.read_csv("data/X_train.csv")
    y_train = pd.read_csv("data/y_train.csv")

    # Train model
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "models/xgb_model.joblib")

    print("Model trained and saved as xgb_model.joblib.")

if __name__ == "__main__":
    train_and_save_model()
