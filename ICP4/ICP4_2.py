import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

train_df=pd.read_csv('./glass.csv')
#print(train_df)

#Checking the number of null vaalues for each column in dataset
#print(train_df.isnull().sum())

#Checking the total number of null values in data set
#print('Total number of null values in data set is : ',train_df.isnull().sum().sum())


train,test=train_test_split(train_df ,test_size=0.4, random_state=0)
X_train=train.drop("Type",axis=1)
Y_train=train['Type']
X_test=test.drop("Type",axis=1)
Y_test=test['Type']
#X_train, X_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.4, random_state=0)
print(len(X_train))

# Fitting Naive Bayes model to the data
model = GaussianNB()
print(Y_train)
# Training the Model on Training Set
model.fit(X_train, Y_train)
# Training the Model on Testing Set
Y_predicted = model.predict(X_test)
#print(Y_predicted)

# Cross Validation compare the predicted and expected values
print(metrics.classification_report(Y_test, Y_predicted))

# Evaluating the Model based on Testing Part
print("Gaussian Model Accuracy is ", metrics.accuracy_score(Y_test, Y_predicted) * 100)


