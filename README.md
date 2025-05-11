# Weather Forecast– MLOps Project

This project predicts, whether it will rain tomorrow or not in Australia.

---

## File Tree
project_mlops/
│
├── data/ # contains raw data and training and test data in CSV file format.
├── evaluation/ # contains evaluation metrics (metrics.json) and evaluation python script
├── models/ # Trained model saved uas joblib file
├── src/ # Contains scripts for data preprocessing, data preparation for model, and model
├── tests/ # Unit tests for training and prediction

---

## How to use?
```bash

### 1. Preprocess and Split
python3 src/split_data.py
### 2. Trian model
python3 src/model.py # used XGBoost Classifier. XGBoost helped dropping minimal data points
### 3. make prediction and evaluation matrices
python3 src/predict.py # change index in the script to see differnt prediction 
python3 evaluation/eval.py
### 4. Unit test
python3 tests/test_train.py
