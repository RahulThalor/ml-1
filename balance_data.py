import pandas as pd
from imblearn.over_sampling import SMOTE

RAW_DATA_PATH = "data/raw/Loan_Default.csv"
BALANCED_DATA_PATH = "data/raw/Loan_Default_balanced.csv"


def load_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    drop_cols = [col for col in ["ID", "Interest_rate_spread"] if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if "Status" in numeric_cols:
        numeric_cols.remove("Status")

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    df = pd.get_dummies(df, drop_first=True)
    return df


def balance_data(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    X = df.drop("Status", axis=1)
    y = df["Status"]

    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)

    balanced_df = pd.DataFrame(X_res, columns=X.columns)
    balanced_df["Status"] = y_res
    return balanced_df


def save_balanced_dataset(df: pd.DataFrame, path: str = BALANCED_DATA_PATH) -> None:
    df.to_csv(path, index=False)


def main() -> None:
    raw_df = load_data()
    processed = preprocess_raw_data(raw_df)
    balanced = balance_data(processed)
    save_balanced_dataset(balanced)

    print("Balanced dataset saved to", BALANCED_DATA_PATH)
    print("Balanced class distribution:")
    print(balanced["Status"].value_counts())


if __name__ == "__main__":
    main()
