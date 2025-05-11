
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import load_and_clean_data, encode_categoricals

def split_and_save():
    df = load_and_clean_data("data/weatherAUS.csv")
    df, _ = encode_categoricals(df)

    X = df.drop(columns=['RainTomorrow'])
    y = df['RainTomorrow']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)

    print("Data split and saved successfully.")

if __name__ == "__main__":
    split_and_save()
