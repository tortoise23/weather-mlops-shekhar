# Weather Forecast– MLOps Project

This project predicts, whether it will rain tomorrow or not in Australia.

---

## File Tree
project_mlops/
│
├── data/ # contains raw data and training and test data in CSV file format. raw data is dvc tracked
├── evaluation/ # contains evaluation metrics (metrics.json) and evaluation python script
├── models/ # Trained model saved as joblib file, which is dvc tracked
├── src/ # Contains scripts for data preprocessing, data preparation for model, and model
├── tests/ # Unit tests for training and prediction
├── .dvc/ # DVC config files
├── .gitignore # Git ignore rules
├── .dvcignore # DVC ignore rules
├── README.md # Project documentation
├── requirements.txt # Python dependencies
└── venv/ # Virtual environment (Git ignored)
---

## How to use?
```bash
### 0. initialization
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
dvc pull

### 1. Preprocess and Split
python3 src/split_data.py
### 2. Trian model
python3 src/model.py # used XGBoost Classifier. XGBoost helped dropping minimal data points
### 3. make prediction and evaluation matrices
python3 src/predict.py # change index in the script to see differnt prediction 
python3 evaluation/eval.py
### 4. Unit test
python3 tests/test_train.py

### 5. MLflow tracking 
phython3 src/model.py

# track with 
mlflow ui
# open http://localhost:5000

```

modified:   README.md
renamed:    src/split_data.py -> src/data_prep_service.py
renamed:    src/model.py -> src/train_service.py
renamed:    src/predict.py -> src/predict_service.py
new file:   src/orchestration_script.py

In Airflow orchestration, we dropped MLflow routines, as it was very resource consuming. See src/services.py

# New Tree
## File Tree
project_mlops/
│
├── .env/ #folder for Airflow
├── dags/ # mlops_pipeline_dag.py for orchestration using Airflow
├── data/ # contains raw data and training and test data in CSV file format. raw data (dvc tracked)
├── evaluation/ # contains evaluation metrics (metrics.json) and evaluation python script
├── models/ # Trained model saved as joblib file (dvc tracked)
├── logs/ # Contains log files for Airflow
├── mlruns/ # Contains logging from MLflow (git ignored)
├── plugins/ # Folder for Airflow
├── src/ #  Data processing, training, and prediction scripts 
│   ├── preprocessing.py
│   ├── data_prep_service.py
│   ├── train_service.py
│   ├── predict_service.py
│   ├── services.py #called by dag/mlops_pipeline_dag.py
│   └── orchestration_script.py           
├── tests/ # Unit tests for training and prediction
├── .dvc/ # DVC config files
├── .gitignore # Git ignore rules
├── .dvcignore # DVC ignore rules
├── README.md # Project documentation
├── requirements.txt # Python dependencies
└── venv/ # Virtual environment (Git ignored)

## Airflow launch instruction
```bash
docker-compose up airflow-init # only for the first launch
docker-compose up -d
# wait for all worker and scheduler becoming healthy. Keep checking staus
docker ps

# once everything is healthy, open ip:8080 in browser --> look for datascientest tag 
#--> look for DAG: weather_data_pipline --> start the DAG --> trigger it to see pipeline working

docker-compose down # to stop the Airflow docker.
```
### Repo links
### github_shekhar: https://github.com/tortoise23/weather-mlops-shekhar
### github_forked: https://github.com/tortoise23/weather_MLOPs_repo
### dagshub_shekhar: https://dagshub.com/tortoise23/weather-mlops-shekhar
### dagshub_forked: https://dagshub.com/tortoise23/weather_MLOPs_repo
