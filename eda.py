import pandas as pd

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
print("Shape:", df.shape)
print("\nTarget counts:\n", df["Churn"].value_counts())

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("\nMissing values after TotalCharges fix:\n", df.isna().sum())

print("\nAverage tenure & monthly charges by Churn:")
print(df.groupby("Churn")[["tenure", "MonthlyCharges"]].mean())

print("\nChurn rate by Contract:")
print(df.groupby("Contract")["Churn"].value_counts(normalize=True))