import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("sensor_Crop_Dataset.csv")

# Features
feature_cols = [
    "Nitrogen", "Phosphorus", "Potassium",
    "Temperature", "Humidity", "pH_Value", "Rainfall"
]

X = df[feature_cols]

def train_and_save_model(target_col, model_file, encoder_file):
    le = LabelEncoder()
    y = le.fit_transform(df[target_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=150,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"\n{target_col} Model Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    joblib.dump(model, model_file, compress=3)
    joblib.dump(le, encoder_file, compress=3)

    print(f"Saved: {model_file}")
    print(f"Saved: {encoder_file}")

# Train Soil Type model
train_and_save_model(
    target_col="Soil_Type",
    model_file="soil_type_model.pkl",
    encoder_file="soil_type_label_encoder.pkl"
)

# Train Variety model
train_and_save_model(
    target_col="Variety",
    model_file="variety_model.pkl",
    encoder_file="variety_label_encoder.pkl"
)

print("\nTraining complete. Models saved.")