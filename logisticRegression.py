import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
x = np.array([100,120,150,170,200,200,202,203,205,210,215,250,270,300,305,310])
y = np.array([1,1,1,1,1,1,1,0,1,0,0,0,0,0,0,0])
plt.scatter(x,y)
plt.title("Pricing Bids")
plt.xlabel("Price")
plt.ylabel("Status (1:Won, 0:Lost)")
#plt.show()
logreg = LogisticRegression(C=1.0,solver = "lbfgs", multi_class = "ovr")
X = x.reshape(-1,1)
logreg.fit(X,y)
print(logreg.predict([[275]]))