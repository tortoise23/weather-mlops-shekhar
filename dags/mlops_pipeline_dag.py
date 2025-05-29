from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from src.services import preprocess_data, train_model, make_prediction


# Define the DAG
with DAG(
    dag_id='weather_data_pipeline',
    tags=['project', 'Datascientest'],
    default_args={
        'owner': 'weather_forecast_team',
        'retries': 1,
        'retry_delay': timedelta(minutes=3)
    },
    description='MLOps workflow DAG for weather forecast project',
    start_date=datetime(2025, 5, 28),
    schedule_interval=None,
    catchup=False
) as dag:

    # Task 1: Data preprocessing service
    preprocess_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data
    )

    # Task 2: Train model service
    train_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model
    )

    # Task 3: Prediction service
    predict_task = PythonOperator(
        task_id='make_prediction',
        python_callable=make_prediction
    )

    preprocess_task >> train_task >> predict_task
