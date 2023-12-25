import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


# Load the dataset
data = pd.read_csv('creditcard.csv')

# Extract features and labels
X = data.drop('Class', axis=1)

# Normalize Data
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Apply KMeans with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42, init='k-means++', n_init=20, max_iter=1000)
kmeans.fit(X)

# Add Cluster Labels to Data
data['Cluster'] = kmeans.labels_
# Find centers of Clusters
centroids = kmeans.cluster_centers_

# Plot the Result
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=kmeans.labels_, cmap='viridis', s=20)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, linewidths=2, color='red', label='Centroids')
plt.title('Clustering Results')
plt.show()