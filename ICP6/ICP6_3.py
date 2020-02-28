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

#Implementing the elbow method to know the ideal number of clusters after scaling
wcss = []
for i in range(2,12):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X_scaled_final)
    wcss.append(kmeans.inertia_)
    score = silhouette_score(X_scaled_final, kmeans.labels_, metric='euclidean')
    print("For n_clusters = {}, silhouette score is {})".format(i, score))
#
plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()