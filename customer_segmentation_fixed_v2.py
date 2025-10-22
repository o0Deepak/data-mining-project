import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Load dataset
data = pd.read_csv("Mall_Customers.csv")

# Encode gender as numeric
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])  # Male=1, Female=0

# Select relevant features
X = data[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA for visualization (2D for clarity)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# -------------------- ELBOW METHOD --------------------
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show(block=False)
plt.pause(2)

# -------------------- PCA VISUALIZATION --------------------
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1])
plt.title('PCA Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show(block=False)
plt.pause(2)

# -------------------- K-MEANS CLUSTERING --------------------
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X[:,2], X[:,3], c=kmeans_labels, cmap='rainbow')
plt.title('K-Means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show(block=False)
plt.pause(2)

data['KMeans_Cluster'] = kmeans_labels

# -------------------- DBSCAN CLUSTERING --------------------
dbscan = DBSCAN(eps=1.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=dbscan_labels, cmap='plasma')
plt.title('DBSCAN Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show(block=False)
plt.pause(2)

data['DBSCAN_Cluster'] = dbscan_labels

# -------------------- AGGLOMERATIVE CLUSTERING --------------------
agg = AgglomerativeClustering(n_clusters=5, linkage='ward')
agg_labels = agg.fit_predict(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=agg_labels, cmap='viridis')
plt.title('Agglomerative Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show(block=False)
plt.pause(2)

data['Agglomerative_Cluster'] = agg_labels

# -------------------- DENDROGRAM --------------------
linked = linkage(X_scaled, method='ward')
plt.figure(figsize=(10,6))
dendrogram(linked)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show(block=False)
plt.pause(2)

# -------------------- CLUSTER CHARACTERISTICS --------------------
print("\nK-Means Cluster Characteristics:")
print(data.groupby('KMeans_Cluster')[['Age','Annual Income (k$)','Spending Score (1-100)']].mean())

print("\nAgglomerative Cluster Characteristics:")
print(data.groupby('Agglomerative_Cluster')[['Age','Annual Income (k$)','Spending Score (1-100)']].mean())

# -------------------- INTERPRETATION --------------------
print("\nSuggested Segment Labels (based on K-Means results):")
print("""
Cluster 0: Low income, low spending (Budget Customers)
Cluster 1: High income, high spending (Premium Customers)
Cluster 2: Average income, average spending (Mid-Tier)
Cluster 3: Low income, high spending (Potential Target)
Cluster 4: High income, low spending (Careful Spenders)
""")

plt.show()
