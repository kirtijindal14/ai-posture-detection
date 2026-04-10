import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load dataset
data = pd.read_csv("posture_dataset.csv")

# Features (landmarks)
X = data.iloc[:, :-1]

# Labels (posture)
y = data.iloc[:, -1]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)

# Evaluation
print("Model Evaluation:")
print(classification_report(y_test, pred))

# Save model
joblib.dump(model, "posture_model.pkl")

print("Model saved as posture_model.pkl")
