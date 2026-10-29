# Sample data sources and notes

The source CSV files were downloaded from the public [mwaskom/seaborn-data repository](https://github.com/mwaskom/seaborn-data):

- `classification/iris.csv` — Iris measurements and species. Source file: [iris.csv](https://github.com/mwaskom/seaborn-data/blob/master/iris.csv). The canonical Iris dataset is attributed to R. A. Fisher (1936), “The use of multiple measurements in taxonomic problems.”
- `classification/penguins.csv` — Palmer penguin measurements, species, island, and sex. Source file: [penguins.csv](https://github.com/mwaskom/seaborn-data/blob/master/penguins.csv). The underlying Palmer Penguins data are associated with Kristen Gorman and Palmer Station, Antarctica LTER; see the [Palmer Penguins project](https://allisonhorst.github.io/palmerpenguins/).
- `regression/auto_mpg.csv` — Auto MPG attributes and fuel economy. Source file: [mpg.csv](https://github.com/mwaskom/seaborn-data/blob/master/mpg.csv). The dataset is also catalogued by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/9/auto+mpg).
- `regression/tips.csv` — restaurant bills and tips. Source file: [tips.csv](https://github.com/mwaskom/seaborn-data/blob/master/tips.csv); the data are documented in the R `reshape2` package.

The files `clustering/iris_unlabeled.csv` and `clustering/penguins_unlabeled.csv` are project copies derived from the corresponding classification CSVs with the `species` column removed. They are provided so the clustering flow uses no target column.

These files are included for educational demonstrations. The upstream repository is a data mirror; check the original dataset terms and attribution requirements before redistributing the datasets publicly.
