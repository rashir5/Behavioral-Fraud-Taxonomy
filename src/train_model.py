from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    os.makedirs("outputs", exist_ok=True)
    joblib.dump(model, "outputs/fraud_model.pkl")

    return model
