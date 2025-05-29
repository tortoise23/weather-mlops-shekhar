import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(path="data/weatherAUS.csv"):
    df = pd.read_csv(path)

    # Drop rows with missing target
    df.dropna(subset=['RainTomorrow'], inplace=True)

    # Fill selected columns (optional)
    df['Rainfall'].fillna(0, inplace=True)
    df['Evaporation'].fillna(df['Evaporation'].median(), inplace=True)
    df['Sunshine'].fillna(df['Sunshine'].median(), inplace=True)

    # Extract features from date
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df.drop(columns=['Date'], inplace=True)

    # Encode target
    df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})

    return df

def encode_categoricals(df):
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna("Missing")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    return df, label_encoders
