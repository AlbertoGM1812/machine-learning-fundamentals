# Machine Learning Fundamentals

A collection of classic machine learning exercises written for two courses (B.Eng. in Artificial Intelligence, ESCOM - IPN): Fundamentals of Artificial Intelligence and Machine Learning. They cover regression, classification, feature selection and clustering with scikit-learn, plus a few algorithms written by hand.

Each folder has its scripts and the small datasets they use, so you can run a script from inside its folder.

## What is inside

- **`regression/`**
  - `ols_vs_sgd_regression.py`: linear and polynomial regression (degrees 1, 2 and 3) solved with ordinary least squares and with stochastic gradient descent, compared by MSE and R2.
  - `california_housing_regression.py`: house price prediction on the California housing data, with polynomial features and different feature scaling methods.
- **`naive-bayes-and-knn/`**
  - `naive_bayes_iris.py` and `naive_bayes_emails.py`: Gaussian and Multinomial Naive Bayes with 5-fold cross-validation, on iris flowers and on spam detection from word counts.
  - `knn_vs_naive_bayes.py`: K-NN with different values of k and weights compared with Naive Bayes.
- **`svm-from-scratch/`**
  - `svm_iris.py`: a one-vs-all linear classifier inspired by SVM, written without any machine learning library. It projects each sample on a vector built from the class centers.
- **`feature-selection-and-knn/`**
  - `feature_selection_rfe.py`: Recursive Feature Elimination with a Random Forest, scored with ROC AUC. It saves the best model.
  - `knn_diabetes.py`: K-NN on the diabetes data using the features selected in the previous step.
- **`kmeans-clustering/`**
  - `kmeans_football.py`: K-Means on the offensive and defensive rating of football clubs.
  - `optimal_k.py`: choosing k with the silhouette score, Calinski-Harabasz, Davies-Bouldin and the elbow method.

Some of the scripts (feature selection, K-NN and K-Means) start from example code given in class.

## How to run

```bash
pip install -r requirements.txt
cd feature-selection-and-knn
python feature_selection_rfe.py     # type "diabetes" or "breast_cancer" when asked for the file name
python knn_diabetes.py              # needs the model saved by the previous script (diabetes)
```

Most scripts ask for the values they need when they start. For `ols_vs_sgd_regression.py`, a learning rate around 0.00001 works well; with 0.001 the degree 3 SGD model diverges because the polynomial features are not scaled. Code comments and messages are in Spanish, as in the original coursework.

The spam dataset `emails.csv` (about 30 MB) is not included. `naive_bayes_emails.py` and `knn_vs_naive_bayes.py` need it in their folder.

## Example result

Gaussian Naive Bayes classifies all 45 iris flowers of the test set correctly (70% train, 30% test).

## Data

- `breast_cancer.csv`: Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository, CC BY 4.0.
- `diabetes.csv`: Pima Indians Diabetes Database, CC0.
- `iris.csv`: Iris dataset, UCI Machine Learning Repository.
- `Clusters_spi_global_rankings.csv`: Soccer Power Index from FiveThirtyEight, CC BY 4.0.
- `cal_housing.csv`: California housing, 1990 US census.
- `datos.csv`: small synthetic dataset from the course assignment.

## Technologies

Python, scikit-learn, pandas, NumPy, Matplotlib, yellowbrick

## Credits

Alberto Gomez Mendez. The feature selection, K-NN and K-Means practices were done together with Fernanda Trujillo Rodriguez and Fabiana Gutierrez Perales.
