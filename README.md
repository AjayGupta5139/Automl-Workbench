# AutoML Workbench

A small, interactive Python project for experimenting with supervised learning and K-Means clustering on CSV files. It is designed as an educational project, not as a fully automatic model-selection system.

## Features

- Reads a CSV file selected by path.
- Supports classification with logistic regression or k-nearest neighbors (KNN).
- Supports regression with linear regression or KNN.
- Supports K-Means clustering when no target column is available.
- Imputes missing values, scales numeric features, and one-hot encodes categorical features.
- Evaluates supervised models on a held-out test split.
- Reports clustering silhouette score and cluster sizes.
- Can predict a value for one new row or display a simple cluster plot.

## Requirements

- Python 3.9 or newer
- See [`requirements.txt`](requirements.txt)

## Setup

Open a terminal in this project folder and run:

```bash
python -m venv .venv
```

Activate the environment:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python automl_workbench.py
```

When asked for a CSV path, enter a path relative to the project folder, such as `data/classification/iris.csv`, or a full path. Use `yes` or `no` when prompted whether a target column is present.

### Example datasets

The included CSVs are in [`data/`](data/):

| Task | Dataset | Suggested target / features |
|---|---|---|
| Classification | `data/classification/iris.csv` | Target: `species`; features: `sepal_length`, `sepal_width`, `petal_length`, `petal_width` |
| Classification | `data/classification/penguins.csv` | Target: `species`; features: `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`, `island`, `sex` |
| Regression | `data/regression/auto_mpg.csv` | Target: `mpg`; features: `cylinders`, `displacement`, `horsepower`, `weight`, `acceleration`, `model_year`, `origin` |
| Regression | `data/regression/tips.csv` | Target: `tip`; features: `total_bill`, `sex`, `smoker`, `day`, `time`, `size` |
| Clustering | `data/clustering/iris_unlabeled.csv` | Use the four sepal/petal measurements; try 3 clusters |
| Clustering | `data/clustering/penguins_unlabeled.csv` | Use the measurement columns (optionally `island` and `sex`); try 3 clusters |

For classification, choose `classification` and then a model name exactly as displayed by the program (for example, `logistic regression` or `knn`). For regression, choose `regression` and either `linear regression` or `knn`.

Clustering files omit the known species column, so the script will treat them as unlabeled examples.

## Metrics

- **Regression:** mean absolute error (MAE), mean squared error (MSE), root mean squared error (RMSE), and R-squared (R²).
- **Classification:** accuracy, weighted precision, weighted recall, and a confusion matrix.
- **Clustering:** silhouette score and number of rows assigned to each cluster.

The clustering silhouette score is an internal measure of cluster separation, not a comparison against known labels. Metrics can vary with the dataset and chosen features; use them as learning aids rather than guarantees of model quality.

## Project layout

```text
Auto_ML/
├── automl_workbench.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    ├── classification/
    ├── regression/
    └── clustering/
```

## Dataset sources

The four source CSVs were downloaded from the public [seaborn-data repository](https://github.com/mwaskom/seaborn-data), which provides datasets for examples. The clustering CSVs are derived from its Iris and Penguins CSVs by removing the `species` column. Dataset-level sources and credits are listed in [`data/README.md`](data/README.md). Please retain those source credits if you redistribute the sample data, and review each dataset's original terms before publishing it.

## Limitations

This is an educational command-line project. It offers a small fixed selection of models, does not search across algorithms or tune hyperparameters, and expects the user to choose features and task type. The held-out split may be unstable on small datasets. It is not intended for high-stakes decisions.
