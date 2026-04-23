import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('data/raw/Loan_Default.csv')

# Drop unnecessary columns
df = df.drop(columns=["ID", "Interest_rate_spread"])

# Identify numeric and categorical columns
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

numeric_cols.remove("Status")

# Fill missing values
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# One-hot encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Separate features and target
X = df.drop("Status", axis=1)
y = df["Status"]

print("Original class distribution:")
print(y.value_counts())

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

print("Balanced class distribution:")
print(pd.Series(y_res).value_counts())

# Combine back to dataframe
df_balanced = pd.DataFrame(X_res, columns=X.columns)
df_balanced['Status'] = y_res

# Save the balanced dataset
df_balanced.to_csv('data/raw/Loan_Default_balanced.csv', index=False)

print("Balanced dataset saved to data/raw/Loan_Default_balanced.csv")