#Loading datasets Using Scikit-Learn

import sklearn
from sklearn import datasets
print(dir(datasets))
housing = datasets.fetch_california_housing()
print(housing.feature_names)
print(housing.data)
print(housing.target_names)
print(housing.DESCR)

from sklearn.datasets import fetch_openml
mice = fetch_openml(name='miceprotein', version=4)
print(mice.details)
