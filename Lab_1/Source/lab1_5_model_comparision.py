import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import  mean_squared_error

#Importing Dataset
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('CC.csv')

#Finding the total number of null values
print('The total null values in the dataset are ',dataset.isnull().sum().sum())
print('-----------------------------------------------------------')

#Eliminating the null values
data = dataset.select_dtypes(include=[np.number]).interpolate().dropna()

#Finding the top 5 most correlated columns to the target variable
numeric_features = data.select_dtypes(include=[np.number])
corr = numeric_features.corr()
print("THE TOP 5 CORRELEATED VALUES ARE")
print(corr['TENURE'].sort_values(ascending=False)[:6])

#Selecting releavent features based on their correlation
train,test=train_test_split(data ,test_size=0.4, random_state=0)
x_train = train.iloc[:,[2,3,-4,-5,-6]]
y_train = train.iloc[:,-1]
x_test = test.iloc[:,[2,3,-4,-5,-6]]
y_test = test.iloc[:,-1]

#Navie Bayes model
model1 = GaussianNB()
model1.fit(x_train, y_train)
y_predicted = model1.predict(x_test)
MSE_Bayes=mean_squared_error(y_test,y_predicted)
print("Mean squared error of Bayes model is : " , MSE_Bayes )

#SVM Model
model2 = svm.SVC()
model2.fit(x_train, y_train)
y_predicted = model2.predict(x_test)
MSE_SVC=mean_squared_error(y_test,y_predicted)
print("Mean squared error of SVM model is : " , MSE_SVC )

#KNN
model3 = KNeighborsClassifier(n_neighbors=5)
model3.fit(x_train, y_train)
y_predicted = model3.predict(x_test)
MSE_KNN=mean_squared_error(y_test,y_predicted)
print("Mean squared error of KNN model is : " , MSE_KNN )

#Suggesting which model is best based on MSE values
if MSE_Bayes>MSE_KNN>MSE_SVC:
    print("SVC classifier is best for use")
elif MSE_KNN>MSE_SVC>MSE_Bayes:
    print("Bayes classifier is best for use")
else:
    print("KNN classifier is best for use")
