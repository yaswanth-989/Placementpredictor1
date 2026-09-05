import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


def load_data():
    df = pd.read_csv(r"D:\2-1 Odd sem\Machine Learning\Placementpredictor\placement_predict_50k Dataset.csv")
    return df


def prepare_data(df):
    target = "PlacementStatus"

    columns_to_remove = [
        "StudentID",
        "Salary Package",
        "IsAnomaly"
    ]

    df = df.drop(columns=columns_to_remove)

    X = df.drop(columns=[target])
    y = df[target]

    label_encoder = {}

    for column in X.select_dtypes(include=["object"]).columns:
        encoder = LabelEncoder()
        X[column] = encoder.fit_transform(X[column])
        label_encoder[column] = encoder

    return X, y, label_encoder


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining records:", len(X_train))
    print("Test records:", len(X_test))

    return X_train, X_test, y_train, y_test


def create_model():
    model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=5,
        random_state=42
    )

    return model


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)

    print("\nModel:")
    print(model)

    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\nAccuracy:", accuracy)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return y_pred


def visualize_tree(model, X):
    plt.figure(figsize=(20, 10))

    tree.plot_tree(
        model,
        feature_names=X.columns,
        class_names=[str(c) for c in model.classes_],
        filled=True
    )

    plt.title("Decision Tree - Placement Prediction")
    plt.show()


def main():
    df = load_data()

    print("Dataset Shape:", df.shape)
    print("\nFirst 5 Records:")
    print(df.head())

    X, y, label_encoder = prepare_data(df)

    print("\nFeatures:")
    print(X.columns.tolist())

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = create_model()

    model = train_model(model, X_train, y_train)

    evaluate_model(model, X_test, y_test)

    visualize_tree(model, X)


if __name__ == "__main__":
    main()