from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import json
import os

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    os.makedirs("outputs", exist_ok=True)

    with open("outputs/metrics.json", "w") as f:
        json.dump(report, f, indent=4)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

    print("Model Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))
