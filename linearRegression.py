import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
data = pd.read_csv("Advertising.csv")
print(data.head())
data.drop('Unnamed: 0', axis=1, inplace=True)
print(data.head())
plt.figure(figsize=(16,8))
plt.scatter(data['TV'], data['Sales'], c="black")
X = data['TV'].values.reshape(-1,1)
Y = data['Sales'].values.reshape(-1,1)
reg = LinearRegression()
reg.fit(X,Y)
predictions = reg.predict(X)
plt.figure(figsize=(16,8))
plt.scatter(
    data['TV'],
    data['Sales'],
    c="black"
)
plt.plot(   
    data['TV'],
    predictions,
    c = "blue",
    linewidth = 2
)
plt.xlabel("Money spent on TV ads ($)")
plt.ylabel("Sales ($)")
plt.show()