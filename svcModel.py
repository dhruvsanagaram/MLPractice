import pandas as pd
total_data = pd.read_csv('Seed_Data.csv')
print(total_data.describe)
X = total_data.iloc[:, 0:7]  #gives total_data w/o target var. column
print(X.describe)
y = total_data.iloc[:, 7]    #gives  total_data w/ only target var. column
print(y.describe)

import sklearn
from sklearn import svm
from sklearn.svm import SVC  #SVC means "Support Vector Classification", and maps data points to a
                                #high-dimensional space so that an optimal hyperplane can be found
                                    #which divides the data into two classes
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=13) #here we split
                                                                                            #the data into 
                                                                                            #80% train and
                                                                                            #20% split

print("Hello")  
#Pre-processing Data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)  #fit will find the mean & std. of train, transform will scale data
                                        #according to these metrics
X_test = sc.transform(X_test)   #we only transform test because if we fit it, then it will transform test
                                    #data using only test data mean & std. which only accounts for 25%
                                    #of the dataset, when we really want to transform test data using train
                                    #data mean & std., which accounts for 75% of the entire dataset
#Data Modelling
classifier = svm.SVC()
classifier.fit(X_train, y_train)   #generate the fit for the model
pred_classifier = classifier.predict(X_test)  #generate the predictions for training set

#Measuring Performance
print(sklearn.metrics.accuracy_score(y_test, pred_classifier))   #compare the predictions to the actual 
                                                                    #outputs to get the accuracy
print(sklearn.metrics.classification_report(y_test, pred_classifier))   #give more info on accuracy



