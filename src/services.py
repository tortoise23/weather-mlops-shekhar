import pandas as pd
import joblib
#import mlflow
#import mlflow.sklearn
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from src.preprocessing import load_and_clean_data, encode_categoricals
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
sys.path.append("/opt/airflow")

def preprocess_data():
    print("Running data preprocessing service...")
    df = load_and_clean_data("/opt/airflow/data/weatherAUS.csv")
    df, _ = encode_categoricals(df)

    X = df.drop(columns=['RainTomorrow'])
    y = df['RainTomorrow']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train.to_csv("/opt/airflow/data/X_train.csv", index=False)
    X_test.to_csv("/opt/airflow/data/X_test.csv", index=False)
    y_train.to_csv("/opt/airflow/data/y_train.csv", index=False)
    y_test.to_csv("/opt/airflow/data/y_test.csv", index=False)

def train_model():
    print("Training model...")
    X_train = pd.read_csv("/opt/airflow/data/X_train.csv")
    y_train = pd.read_csv("/opt/airflow/data/y_train.csv")

    model = xgb.XGBClassifier(
        use_label_encoder=False, 
        eval_metric='logloss',
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100
    )
    
    model.fit(X_train, y_train)

    joblib.dump(model, "/opt/airflow/models/xgb_model.joblib")

    y_pred = model.predict(X_train)
    acc = accuracy_score(y_train, y_pred)
    f1 = f1_score(y_train, y_pred)
    print(f"Training Accuracy: {acc:.4f}")
    print(f"Training F1 Score: {f1:.4f}")

def make_prediction():
    print("Making predictions...")
    model = joblib.load("/opt/airflow/models/xgb_model.joblib")
    X_test = pd.read_csv("/opt/airflow/data/X_test.csv")
    y_test = pd.read_csv("/opt/airflow/data/y_test.csv")

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Prediction accuracy: {accuracy:.2%}")