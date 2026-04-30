import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data(path):
    # Load data
    df = pd.read_csv(path)

    # Drop null values
    df = df.dropna()

    # Separate features and target
    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]

    # Convert categorical to numeric
    X = pd.get_dummies(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Handle imbalance
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test
