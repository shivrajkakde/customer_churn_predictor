import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.drop(columns=["customerID"])

y = (df["Churn"] == "Yes").astype(int)
X = df.drop(columns=["Churn"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

cat_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", cat_transformer, cat_cols),
    ]
)

X_train_encoded = preprocessor.fit_transform(X_train)
print("Encoded train shape:", X_train_encoded.shape)





from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", num_transformer, num_cols),
    ("cat", cat_transformer, cat_cols),
])

clf = Pipeline(steps=[
    ("prep", preprocessor),
    ("model", LogisticRegression(max_iter=1000)),
])

clf.fit(X_train, y_train)
probs = clf.predict_proba(X_test)[:, 1]
preds = (probs >= 0.5).astype(int)

print("Baseline ROC-AUC:", round(roc_auc_score(y_test, probs), 4))
print(classification_report(y_test, preds))



from xgboost import XGBClassifier

xgb_clf = Pipeline(steps=[
    ("prep", preprocessor),
    ("model", XGBClassifier(eval_metric="logloss", random_state=42)),
])

xgb_clf.fit(X_train, y_train)
xgb_probs = xgb_clf.predict_proba(X_test)[:, 1]
print("XGBoost ROC-AUC:", round(roc_auc_score(y_test, xgb_probs), 4))




import joblib

# Save whichever pipeline performed better (e.g., xgb_clf or clf)
joblib.dump(xgb_clf, "churn_model.joblib")
print("Saved model to churn_model.joblib")