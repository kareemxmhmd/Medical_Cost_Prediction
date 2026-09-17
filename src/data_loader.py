import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "insurance.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print(df.dtypes)
    print(df["charges"].describe())
