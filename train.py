import numpy as np
import pandas as pd
import pickle

# Load dataset (FIXED PATH)
dataset = pd.read_csv("upi_fraud_dataset.csv")

# Drop unwanted columns
dataset = dataset.drop(['Id', 'upi_number'], axis=1)

# 🔍 Check label distribution
print("Fraud Risk Distribution:")
print(dataset['fraud_risk'].value_counts())

# Split features & target
X = dataset.drop('fraud_risk', axis=1)
y = dataset['fraud_risk']

# Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(random_state=42)
}

from sklearn.metrics import classification_report

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    report = classification_report(y_test, pred, output_dict=True)

    # 🔥 Use weighted avg (NOT class "1")
    results[name] = {
        "model": model,
        "accuracy": report["accuracy"],
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"]
    }

# 🔥 Better model selection (balanced)
best_model_name = max(results, key=lambda x: results[x]["f1"])
best_model = results[best_model_name]["model"]

print("Best Model:", best_model_name)

# Save files
pickle.dump(models, open("models.pkl", "wb"))
pickle.dump(best_model_name, open("best_model_name.pkl", "wb"))
pickle.dump(results, open("results.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
