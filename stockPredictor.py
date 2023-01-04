import yfinance as yf
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
DATA_PATH = "msft_data.json"

if os.path.exists(DATA_PATH):
    with open(DATA_PATH) as f:
        msft_hist = pd.read_json(DATA_PATH)
else:
    msft = yf.Ticker("MSFT")
    msft_hist = msft.history(period="max")

    msft_hist.to_json(DATA_PATH)

#print(msft_hist.head(10))
#plt.plot(msft_hist["Close"])
#plt.show()

data = msft_hist[["Close"]]
data = data.rename(columns = {'Close':'Actual_Close'})
data["Target"] = msft_hist.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]
#print(data.head(5))
msft_prev = msft_hist.copy()
msft_prev = msft_prev.shift(1)
#print("\n")
#print(msft_prev.head(5))
predictors = ["Close", "Volume", "Open", "High", "Low"]
data = data.join(msft_prev[predictors]).iloc[1:]
print(data.head(5))

from sklearn.ensemble import RandomForestClassifier
import numpy as np
model = RandomForestClassifier(n_estimators = 100, min_samples_split=200, random_state=1)
train = data.iloc[:-100]
test = data.iloc[-100:]
model.fit(train[predictors], train["Target"])

from sklearn.metrics import precision_score

preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)
print(precision_score(test["Target"], preds))

combined = pd.concat({"Target": test["Target"],"Predictions": preds}, axis=1)
#plt.plot(combined)
#plt.show()

def backtest(data,model,predictors,start=1000,step=750):
    predictions=[]
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()

        model.fit(train[predictors], train["Target"])

        preds = model.predict_proba(test[predictors])[:,1]
        preds = pd.Series(preds, index=test.index)
        preds[preds > 0.7125] = 1
        preds[preds <= 0.7125] = 0

        combined = pd.concat({"Target": test["Target"], "Predictions": preds}, axis=1)
        predictions.append(combined)
    return pd.concat(predictions)

predictions = backtest(data,model,predictors)
print(predictions["Predictions"].value_counts())
print(precision_score(predictions["Target"], predictions["Predictions"]))
