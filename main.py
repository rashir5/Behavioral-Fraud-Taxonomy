from src.data_preprocessing import load_and_preprocess_data
from src.train_model import train_model
from src.evaluate_model import evaluate_model
from src.explain_model import explain_model

DATA_PATH = "data/sample_transactions.csv"

def main():
    X_train, X_test, y_train, y_test = load_and_preprocess_data(DATA_PATH)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    explain_model(model, X_test)

if __name__ == "__main__":
    main()
