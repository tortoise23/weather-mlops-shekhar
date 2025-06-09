import evidently
import datetime
import pandas as pd
import numpy as np
import requests
import zipfile
import io
import json
import os, sys
import xgboost as xgb

from sklearn import datasets, ensemble, model_selection
from sklearn.preprocessing import LabelEncoder
from scipy.stats import anderson_ksamp

from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from evidently.ui.workspace import Workspace

# ignore warnings
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

# Append the absolute path of the src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../data")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

RAW_DATA_PATH = "../data/weatherAUS.csv"
def load_data():
    df = pd.read_csv(RAW_DATA_PATH, parse_dates=["Date"])
    df = df.sort_values("Date")
    # Drop rows with missing target
    df.dropna(subset=['RainTomorrow'], inplace=True)
    target = 'RainTomorrow'
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = [
        col for col in df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        if col != target
    ]
    return df, numerical_features, categorical_features
numerical_features, categorical_features = load_data()[1:3]

def clean_data():
    # Fill selected columns (optional)
    df, _, _ = load_data()
    df['Rainfall'].fillna(0, inplace=True)
    df['Evaporation'].fillna(df['Evaporation'].median(), inplace=True)
    df['Sunshine'].fillna(df['Sunshine'].median(), inplace=True)

    # Extract features from date
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df1 = df.copy()
    df.drop(columns=['Date'], inplace=True)
    # Encode target
    df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})
    return df#, df1

def encode_categoricals():
    #df, df1 = clean_data()
    df = clean_data()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna("Missing")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    #df['Date'] = df1['Date']
    return df

def get_year_data_ref():
    df = encode_categoricals()
    df1 = df[(df['Year'] == 2008) | (df['Year'] == 2009) | (df['Year'] == 2010)]  # Reference year data
    return df1

def get_year_data(year):
    df = encode_categoricals()
    df1 = df[df['Year'] == year]
    return df1


def reference_data():
    df_ref = get_year_data_ref()  # Reference year data
    X = df_ref.drop(columns=['RainTomorrow'])
    y = df_ref['RainTomorrow']
    return X, y

def current_data(year):
    df_current = get_year_data(year)
    X = df_current.drop(columns=['RainTomorrow'])
    y = df_current['RainTomorrow']
    return X, y

def model_training():
    # Reference and current data split
    X_train, X_test, y_train, y_test = model_selection.train_test_split(reference_data()[0],reference_data()[1],test_size=0.2, random_state=42)
    # Model training
    classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss',max_depth=6,learning_rate=0.1,n_estimators=100)
    classifier.fit(X_train, y_train)

    # Predictions
    preds_train = classifier.predict(X_train)
    preds_test = classifier.predict(X_test)

    # Add actual target and prediction columns to the training data for later performance analysis
    X_train['target'] = y_train
    X_train['prediction'] = preds_train

    # Add actual target and prediction columns to the test data for later performance analysis
    X_test['target'] = y_test
    X_test['prediction'] = preds_test

    return X_train, X_test, y_train, y_test, classifier

def model_validation(X_train, X_test, y_train, y_test):    
    column_mapping = ColumnMapping()
    column_mapping.target = "target"
    column_mapping.prediction = "prediction"
    column_mapping.numerical_features = numerical_features
    column_mapping.categorical_features = categorical_features

    classification_performance_report = Report(metrics=[ClassificationPreset()], name="Model Validation Report", tags=["validation", "performance"])
    classification_performance_report.run(reference_data=X_train.sort_index(), 
                                    current_data=X_test.sort_index(),
                                    column_mapping=column_mapping
                                    )
    return classification_performance_report

def prod_model_drift(classifier, X_ref, y_ref):
    column_mapping = ColumnMapping()
    column_mapping.target = "target"
    column_mapping.prediction = "prediction"
    column_mapping.numerical_features = numerical_features
    column_mapping.categorical_features = categorical_features

    ref_data = X_ref.copy()
    ref_data["target"] = y_ref
    ref_data["prediction"] = classifier.predict(X_ref)

    classification_performance_report = Report(metrics=[ClassificationPreset()], name="Production Model Drift Report", tags=["production", "drift"])
    classification_performance_report.run(
        reference_data=None,
        current_data=ref_data,
        column_mapping=column_mapping
    )

    return classification_performance_report


def prod_model_drift_month(classifier,year):
    column_mapping = ColumnMapping()
    column_mapping.target = "target"
    column_mapping.prediction = "prediction"
    column_mapping.numerical_features = numerical_features
    column_mapping.categorical_features = categorical_features

    X_ref, y_ref = reference_data()
    ref_data = X_ref.copy()
    ref_data["target"] = y_ref
    ref_data["prediction"] = classifier.predict(X_ref)

    X_cur, y_cur = current_data(year)
    cur_data = X_cur.copy()
    cur_data["target"] = y_cur
    cur_data["prediction"] = classifier.predict(X_cur)

    month_report = Report(metrics=[ClassificationPreset()], name=f"Monthly Model Drift Report - Year {year}", tags=[f"year_{year}", "drift"])
    month_report.run(
        reference_data=ref_data,
        current_data=cur_data,
        column_mapping=column_mapping
    )

    return month_report

def target_drift(year):
    column_mapping_drift = ColumnMapping()
    column_mapping_drift.target = "target"
    column_mapping_drift.prediction = "prediction"
    column_mapping_drift.numerical_features = numerical_features
    column_mapping_drift.categorical_features = []

    X_ref, y_ref = reference_data()
    ref_data = X_ref.copy()
    ref_data["target"] = y_ref

    X_cur, y_cur = current_data(year)
    cur_data = X_cur.copy()
    cur_data["target"] = y_cur

    data_drift_report = Report(metrics=[DataDriftPreset()],name="Data Drift Report", tags=[f"year_{year}", "data drift"])
    data_drift_report.run(
        reference_data=ref_data,
        current_data=cur_data,
        column_mapping=column_mapping_drift,
    )
    return data_drift_report

def add_report(workspace, project_name, project_description, report):
    
    # Check if project already exists
    project = None
    for p in workspace.list_projects():
        if p.name == project_name:
            project = p
            break

    # Create a new project if it doesn't exist
    if project is None:
        project = workspace.create_project(project_name)
        project.description = project_description
    # Add report to the project
    workspace.add_report(project.id, report)
    
    print(f"New item added to project {project_name}")

if __name__ == "__main__":
    WORKSPACE_NAME = "workspace"
    PROJECT_NAME = "Yearly Drift Analysis of weather Data_with 3 years training"
    PROJECT_DESCRIPTION = "Evidently Dashboards"

    # Prepare reference data only once
    X_ref, y_ref = reference_data()

    # Train model
    X_train, X_test, y_train, y_test, classifier = model_training()

    # Create workspace
    workspace = Workspace(WORKSPACE_NAME)

    # Model validation report
    classification_performance_report = model_validation(X_train, X_test, y_train, y_test)
    add_report(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, classification_performance_report)

    # Production model drift report
    prod_model_drift_report = prod_model_drift(classifier, X_ref, y_ref)
    add_report(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, prod_model_drift_report)

    # Yearly reports
    for year in np.arange(2011,2017,1):  # From 2011 to 2016
        model_drift_report_year = prod_model_drift_month(classifier, year)
        add_report(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, model_drift_report_year)

        # Data drift report
        data_drift_report = target_drift(year)
        add_report(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, data_drift_report)
