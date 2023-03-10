from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler     #X_std = (X - X.min)/(X.max - X.min)
                                                        #X_scaled = X_std * (max - min) + min
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
iris = load_iris()  
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.2, random_state=42)
pipe_lr = Pipeline([('minmax', MinMaxScaler()), ('lr', LogisticRegression())])
pipe_lr.fit(X_train, y_train)
score = pipe_lr.score(X_test, y_test)
print(score)

