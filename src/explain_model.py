import shap
import matplotlib.pyplot as plt
import os

def explain_model(model, X_test):
    os.makedirs("outputs", exist_ok=True)

    sample = X_test.sample(
        min(100, len(X_test)),
        random_state=42
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    shap.summary_plot(shap_values, sample, show=False)
    plt.savefig("outputs/shap_summary.png", bbox_inches="tight")
    plt.close()

    print("SHAP explanation saved successfully.")
