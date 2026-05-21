import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("sensor_Crop_Dataset.csv")

# Features
feature_cols = [
    "Nitrogen", "Phosphorus", "Potassium",
    "Temperature", "Humidity", "pH_Value", "Rainfall"
]

X = df[feature_cols]

# ----------------------------
# 1. Soil Type Model
# ----------------------------
soil_le = LabelEncoder()
y_soil = soil_le.fit_transform(df["Soil_Type"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y_soil, test_size=0.2, random_state=42, stratify=y_soil
)

soil_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)
soil_model.fit(X_train, y_train)

soil_pred = soil_model.predict(X_test)
print("Soil Type Model Accuracy:", accuracy_score(y_test, soil_pred))
print(classification_report(y_test, soil_pred, target_names=soil_le.classes_))

joblib.dump(soil_model, "soil_type_model.pkl")
joblib.dump(soil_le, "soil_type_label_encoder.pkl")

# ----------------------------
# 2. Variety Model
# ----------------------------
var_le = LabelEncoder()
y_var = var_le.fit_transform(df["Variety"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y_var, test_size=0.2, random_state=42, stratify=y_var
)

var_model = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    class_weight="balanced"
)
var_model.fit(X_train, y_train)

var_pred = var_model.predict(X_test)
print("Variety Model Accuracy:", accuracy_score(y_test, var_pred))
print(classification_report(y_test, var_pred, target_names=var_le.classes_))

joblib.dump(var_model, "variety_model.pkl")
joblib.dump(var_le, "variety_label_encoder.pkl")

print("Training complete. Models saved.")