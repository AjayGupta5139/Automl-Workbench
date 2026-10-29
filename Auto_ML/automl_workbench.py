import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def ask_yes_no(prompt):
    while True:
        answer = input(prompt).strip().lower()
        if answer in {"yes", "y"}:
            return True
        if answer in {"no", "n"}:
            return False
        print("Please enter yes or no.")


def get_columns(prompt, dataframe):
    columns = [column.strip() for column in input(prompt).split(",") if column.strip()]
    invalid = [column for column in columns if column not in dataframe.columns]
    if invalid:
        raise ValueError(f"Invalid column(s): {', '.join(invalid)}")
    if not columns:
        raise ValueError("Select at least one column.")
    return columns


def build_preprocessor(features):
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]

    numeric_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    categorical_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"),
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


def run_clustering(dataframe):
    feature_columns = get_columns("Enter input feature(s), separated by commas: ", dataframe)
    features = dataframe[feature_columns].copy()
    if len(features) < 3:
        raise ValueError("Clustering needs at least 3 rows to calculate a silhouette score.")

    transformed = build_preprocessor(features).fit_transform(features)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    max_clusters = len(features) - 1
    try:
        cluster_count = int(input(f"Number of clusters (2-{max_clusters}): "))
    except ValueError as error:
        raise ValueError("Enter a whole number for the cluster count.") from error
    if not 2 <= cluster_count <= max_clusters:
        raise ValueError(f"Cluster count must be between 2 and {max_clusters}.")

    model = KMeans(n_clusters=cluster_count, random_state=0, n_init=10)
    labels = model.fit_predict(transformed)
    dataframe["Cluster"] = labels
    print("\nClustering evaluation")
    print(f"Silhouette score: {silhouette_score(transformed, labels):.4f}")
    print("Cluster sizes:")
    print(pd.Series(labels).value_counts().sort_index().to_string())

    if ask_yes_no("Show a cluster graph? (yes/no): "):
        if transformed.shape[1] >= 2:
            x_values, y_values = transformed[:, 0], transformed[:, 1]
            x_label, y_label = "Transformed feature 1", "Transformed feature 2"
        else:
            x_values, y_values = range(len(labels)), transformed[:, 0]
            x_label, y_label = "Row", "Transformed feature"
        plt.scatter(x_values, y_values, c=labels, cmap="viridis")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title("K-Means Clustering")
        plt.show()


def run_supervised(dataframe):
    feature_columns = get_columns(
        "Enter input feature(s), separated by commas: ", dataframe
    )
    target_column = input("Enter the target column: ").strip()
    if target_column not in dataframe.columns:
        raise ValueError(f"Invalid target column: {target_column}")
    if target_column in feature_columns:
        raise ValueError("The target column cannot also be an input feature.")

    task = input("Choose task (regression/classification): ").strip().lower()
    if task not in {"regression", "classification"}:
        raise ValueError("Task must be regression or classification.")

    usable = dataframe[feature_columns + [target_column]].dropna(subset=[target_column])
    features = usable[feature_columns].copy()
    target = usable[target_column].copy()
    if task == "regression":
        target = pd.to_numeric(target, errors="coerce")
        valid_rows = target.notna()
        features, target = features.loc[valid_rows], target.loc[valid_rows]
        if len(features) < 5:
            raise ValueError("At least 5 rows with numeric targets are required for a useful test split.")
        models = {
            "linear regression": LinearRegression(),
            "knn": KNeighborsRegressor(n_neighbors=min(5, max(1, int(len(features) * 0.2)))),
        }
    else:
        if target.nunique() < 2:
            raise ValueError("Classification needs at least two target classes.")
        if len(features) < 4:
            raise ValueError("At least 4 rows are required for classification.")
        models = {
            "logistic regression": LogisticRegression(max_iter=1000),
            "knn": KNeighborsClassifier(n_neighbors=min(5, max(1, int(len(features) * 0.2)))),
        }

    model_name = input(f"Choose model ({'/'.join(models)}): ").strip().lower()
    if model_name not in models:
        raise ValueError(f"Choose one of: {', '.join(models)}")

    stratify = target if task == "classification" and target.value_counts().min() >= 2 else None
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )
    except ValueError as error:
        raise ValueError(f"Could not create a train/test split: {error}") from error

    pipeline = make_pipeline(build_preprocessor(x_train), models[model_name])
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    print("\nEvaluation metrics (held-out test data)")
    if task == "regression":
        print(f"Mean absolute error (MAE): {mean_absolute_error(y_test, predictions):.4f}")
        print(f"Mean squared error (MSE): {mean_squared_error(y_test, predictions):.4f}")
        print(f"Root mean squared error (RMSE): {mean_squared_error(y_test, predictions) ** 0.5:.4f}")
        print(f"R-squared (R²): {r2_score(y_test, predictions):.4f}")
    else:
        print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
        print(f"Precision (weighted): {precision_score(y_test, predictions, average='weighted', zero_division=0):.4f}")
        print(f"Recall (weighted): {recall_score(y_test, predictions, average='weighted', zero_division=0):.4f}")
        print(f"Confusion matrix:\n{confusion_matrix(y_test, predictions)}")

    if ask_yes_no("Enter a new row to get a prediction? (yes/no): "):
        new_row = {}
        for column in feature_columns:
            new_row[column] = input(f"Enter {column}: ")
        prediction = pipeline.predict(pd.DataFrame([new_row]))[0]
        print(f"Prediction: {prediction}")


def main():
    csv_path = input("Enter the path of your CSV file: ").strip().strip('"')
    dataframe = pd.read_csv(csv_path)
    print(dataframe.head())

    try:
        if ask_yes_no("Do you have a target column? (yes/no): "):
            run_supervised(dataframe)
        else:
            run_clustering(dataframe)
    except (ValueError, KeyError) as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
