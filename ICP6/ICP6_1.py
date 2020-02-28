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

#Implementing the elbow method to know the ideal number of clusters
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

#Plotting the elbow curve
plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()