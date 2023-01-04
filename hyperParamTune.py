import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
iris = load_iris()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.2, random_state=29)
classifier = RandomForestClassifier(n_estimators=2, min_samples_split=3, min_samples_leaf=2) #Generating RFC
print(classifier.fit(X_train, y_train))   #Fitting the RFC based on the training set
pred_clf = classifier.predict(X_test)
print(sklearn.metrics.accuracy_score(y_test, pred_clf))   #Finding accuracy score of 93%
from sklearn.model_selection import GridSearchCV    #GridSearchCV can find the best params to increase accuracy
param_grid = {'n_estimators':[2,5,10,20], 'min_samples_split':[2,3], 'min_samples_leaf':[1,2,3]}
grid_search = GridSearchCV(estimator=classifier, param_grid = param_grid)
print(grid_search.fit(X_train, y_train))  #fitting grid_search
print(grid_search.best_params_)     #We can see that the best params are min_samples_leaf = 3
                                        #min_samples_split = 2, n_estimators = 5

#We can change the RFC to these new parameters
classifier = RandomForestClassifier(n_estimators=20, min_samples_split=3, min_samples_leaf=1)
print(classifier.fit(X_train, y_train))
pred_clf = classifier.predict(X_test)
print(sklearn.metrics.accuracy_score(y_test, pred_clf))




