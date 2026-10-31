import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler

# Path built from this file's location, so it works from anywhere
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "creditcard.csv"
TARGET = "Class"
SCALE_COLS = ["Time", "Amount", "Hour"]


def load_data(path=DATA_PATH, drop_dupes=True, add_hour=True, verbose=True):
    """Load the CSV and apply the safe cleaning steps."""
    df = pd.read_csv(path)
    if verbose:
        print("Missing values:", df.isnull().sum().sum())
    if drop_dupes:
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        if verbose:
            print(f"Removed {before - len(df)} duplicates")
    if add_hour:
        df["Hour"] = (df["Time"] // 3600) % 24
    return df


def get_train_test(test_size=0.2, random_state=42, verbose=True, **load_kwargs):
    """Return X_train, X_test, y_train, y_test (stratified, reproducible)."""
    df = load_data(verbose=verbose, **load_kwargs)
    X = df.drop(columns=TARGET)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    if verbose:
        print("Train:", X_train.shape, f"fraud = {y_train.mean()*100:.3f}%")
        print("Test :", X_test.shape, f"fraud = {y_test.mean()*100:.3f}%")
    return X_train, X_test, y_train, y_test


def make_preprocessor(scale_cols=SCALE_COLS):
    """Scaler for Time/Amount/Hour. Use it INSIDE a Pipeline."""
    return ColumnTransformer(
        [("scale", RobustScaler(), scale_cols)],
        remainder="passthrough"
    )