import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# بيانات تدريب بسيطة
data = {
    "temperature": [30, 40, 25, 35, 28],
    "humidity": [70, 30, 90, 50, 75],
    "moisture": [40, 20, 80, 30, 60],
    "ph": [6.5, 5.5, 7.0, 6.0, 6.8],
    "status": ["Healthy", "Dry", "Fungus", "Dry", "Healthy"]
}

df = pd.DataFrame(data)

X = df[["temperature","humidity","moisture","ph"]]
y = df["status"]

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "turf_model.pkl")

print("Model created successfully!")