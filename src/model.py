import pandas as pd
import xgboost as xgb
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score

def train_and_save_model():
    # Load preprocessed data
    X_train = pd.read_csv("data/X_train.csv")
    y_train = pd.read_csv("data/y_train.csv")

    # Train model with MLflow tracking
    with mlflow.start_run():
        
        model = xgb.XGBClassifier(
            use_label_encoder=False, 
            eval_metric='logloss',
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100
        )
    
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "models/xgb_model.joblib")

    mlflow.sklearn.log_model(model, "model")
    mlflow.log_param("max_depth", 6)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("n_estimators", 100)

    y_pred = model.predict(X_train)
    acc = accuracy_score(y_train, y_pred)
    f1 = f1_score(y_train, y_pred)

    mlflow.log_metric("train_accuracy", acc)
    mlflow.log_metric("train_f1_score", f1)

    
    
    print("Model trained, logged with MLflow, and saved locally.")

if __name__ == "__main__":
    train_and_save_model()
