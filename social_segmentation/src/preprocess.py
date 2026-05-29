import pandas as pd

def clean_data(df):
    df = df.copy()
    df = df.drop_duplicates()
    df = df.dropna()

    numeric_cols = [c for c in df.columns if c != "id"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()
    return df