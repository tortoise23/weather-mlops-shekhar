import joblib
import pandas as pd
import sys
import os
# Append the absolute path of the src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from src.train_service import train_and_save_model


def test_model_training():
    # Train and save model
    train_and_save_model()

    # Load and check model
    model = joblib.load("models/xgb_model.joblib")
    assert hasattr(model, "predict"), "Model does not have predict method."

    print("test_model_training passed.")

if __name__ == "__main__":
    test_model_training()
