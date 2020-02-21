import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
dataset = pd.read_csv('winequality-red.csv')

#Prints the total number of null values in the dataset
print('The total null values in the dataset are ',dataset.isnull().sum().sum())
print('-----------------------------------------------------------')

X=dataset.drop(['quality'],axis=1)
Y=dataset[['quality']]

# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, Y)

#predicting the values
y_pred=regr.predict(X)

#Evaluating the model
print("Variance score: %.2f" % r2_score(Y,y_pred))
print("Mean squared error: %.2f" % mean_squared_error(Y,y_pred))
print('-----------------------------------------------------------')

#Calaculating the top 3 correlated variables
numeric_features = dataset.select_dtypes(include=[np.number])
corr = numeric_features.corr()
print('Top 3 correlated variables to the target varialble quality is: ')
print(corr['quality'].sort_values(ascending=False)[:3],'\n')

