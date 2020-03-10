import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

dataset = pd.read_csv('CC.csv')

#Finding the total number of null values
null_values = pd.DataFrame(dataset.isnull().sum().sort_values(ascending=False)[:25])
print('The total null values in the dataset are ',dataset.isnull().sum().sum())
print(null_values)
print('-----------------------------------------------------------')

#Eliminating the null values
data = dataset.select_dtypes(include=[np.number]).interpolate().dropna()

#Finding the top 5 most correlated columns to the target variable
numeric_features = data.select_dtypes(include=[np.number])
corr = numeric_features.corr()
print("THE TOP 5 CORRELEATED VALUES ARE")
print(corr['TENURE'].sort_values(ascending=False)[:6])

# Preprocessing the data
data_preprocess = preprocessing.StandardScaler()
data_preprocess.fit(data)
cleaned_data = data_preprocess.transform(data)
cleaned_data_final = pd.DataFrame(cleaned_data, columns = data.columns)

#assigning data to the independent variable
x = cleaned_data_final.iloc[:,[2,3,-4,-5,-6]]
y = cleaned_data_final.iloc[:,-1]

#Implementing the elbow method to know the ideal number of clusters
wcss = []
for i in range(2,12):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    sh_score = silhouette_score(x, kmeans.labels_, metric='euclidean')
    print("For n_clusters = {}, silhouette score is {})".format(i, sh_score))

#Plotting the elbow curve
plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()
