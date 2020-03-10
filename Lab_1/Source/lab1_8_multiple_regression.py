import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import  mean_squared_error
import seaborn as sns; sns.set(color_codes=True)
from sklearn import linear_model

#Importing Dataset
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('CC.csv')

#Finding the total number of null values
print('The total null values in the dataset are ',dataset.isnull().sum().sum())
print('Column wise Null values are:')
print(dataset.isnull().sum())
print('-----------------------------------------------------------')


#Eliminating the null values
data = dataset.select_dtypes(include=[np.number]).interpolate().dropna()

#Finding the top 5 most correlated columns to the target variable
numeric_features = data.select_dtypes(include=[np.number])
plt.figure(figsize=(20,20))
corr = numeric_features.corr()
print("THE TOP 5 CORRELEATED VALUES ARE")
sns.heatmap(corr, annot=True, cmap ='viridis')
plt.show()
print(corr['TENURE'].sort_values(ascending=False)[:6])

#Analyzing data after eliminating null values
Train,Test=train_test_split(data ,test_size=0.4, random_state=0)
X_train = Train.iloc[:,[2,3,-4,-5,-6]]
Y_train = Train.iloc[:,-1]
X_test = Test.iloc[:,[2,3,-4,-5,-6]]
Y_test = Test.iloc[:,-1]
lr = linear_model.LinearRegression()
model = lr.fit(X_train, Y_train)
r_square_error=lr.score(X_test, Y_test)
print ("R square is: \n", r_square_error)
Y_pred = lr.predict(X_test)
rmse=mean_squared_error(Y_test, Y_pred)
print ('RMSE is: \n',rmse)

#
plt.figure(figsize=(20,20))
plt.scatter(Y_pred, Y_test, alpha=.75,
            color='r') #alpha helps to show overlapping data
plt.xlabel('Predicted Values ')
plt.ylabel('Actual values')
plt.title('Linear Regression Model')
plt.show()