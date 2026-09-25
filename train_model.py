import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ---------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------

df = pd.read_csv("Customer Churn.csv")

print("Original dataset shape:", df.shape)


# ---------------------------------------------------
# 2. REMOVE DUPLICATES
# ---------------------------------------------------

df = df.drop_duplicates()

print("Dataset shape after removing duplicates:", df.shape)


# ---------------------------------------------------
# 3. DEFINE FEATURES AND TARGET
# ---------------------------------------------------

# We deliberately exclude:
# - Churn → target variable
# - Status → possible data leakage

features = [
    "Call  Failure",
    "Complains",
    "Subscription  Length",
    "Charge  Amount",
    "Seconds of Use",
    "Frequency of use",
    "Frequency of SMS",
    "Distinct Called Numbers",
    "Age Group",
    "Tariff Plan",
    "Age",
    "Customer Value"
]

X = df[features]
y = df["Churn"]


# ---------------------------------------------------
# 4. TRAIN-TEST SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------
# 5. BUILD LOGISTIC REGRESSION PIPELINE
# ---------------------------------------------------

model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic_regression", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])


# ---------------------------------------------------
# 6. TRAIN MODEL
# ---------------------------------------------------

model.fit(X_train, y_train)


# ---------------------------------------------------
# 7. MAKE PREDICTIONS
# ---------------------------------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# ---------------------------------------------------
# 8. MODEL EVALUATION
# ---------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\nMODEL PERFORMANCE")
print("-------------------------")
print(f"Accuracy  : {accuracy:.3f}")
print(f"Precision : {precision:.3f}")
print(f"Recall    : {recall:.3f}")
print(f"F1 Score  : {f1:.3f}")
print(f"ROC-AUC   : {auc:.3f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ---------------------------------------------------
# 9. SAVE MODEL
# ---------------------------------------------------

joblib.dump(model, "churn_model.pkl")

# Save feature information separately
joblib.dump(features, "feature_info.pkl")

print("\nModel saved as churn_model.pkl")
print("Feature information saved as feature_info.pkl")
