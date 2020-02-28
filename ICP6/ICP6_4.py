import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
import seaborn as sns

#Importing Dataset
dataset = pd.read_csv('CC.csv')

#Finding the top 5 most correlated columns to the target variable
numeric_features = dataset.select_dtypes(include=[np.number])
corr = numeric_features.corr()
#print(corr)
print(corr['TENURE'].sort_values(ascending=False)[:6])

#Eliminating the null values
data = dataset.select_dtypes(include=[np.number]).interpolate().dropna()
#print(data)

#assigning data to the independent variable
x = data.iloc[:,[2,3,-4,-5,-6]]
y = data.iloc[:,-1]

#Scaling the data
scaler = preprocessing.StandardScaler()
scaler.fit(x)
X_scaled_initial = scaler.transform(x)
X_scaled_final = pd.DataFrame(X_scaled_initial, columns = x.columns)

pca = PCA(2)
x_pca = pca.fit_transform(X_scaled_final)
df1 = pd.DataFrame(data=x_pca)
df_final = pd.concat([df1,dataset[['TENURE']]],axis=1)
print(df_final)