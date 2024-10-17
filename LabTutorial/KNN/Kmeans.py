import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.neighbors import KNeighborsClassifier # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
from sklearn.cluster import KMeans # type: ignore
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore

file_path = 'concrete_data.csv'
data = pd.read_csv(file_path)

X = data.drop(columns=['Strength'])
y = data['Strength']

y_binned = pd.cut(y, bins=3, labels=[0, 1, 2])

X_train, X_test, y_train, y_test = train_test_split(X, y_binned, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f'KNN Classifier Accuracy: {accuracy * 100:.2f}%')
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_train_scaled)
plt.figure(figsize=(12, 6))
plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, cmap='viridis', marker='o', edgecolor='k', alpha=0.6, label='Actual Strength')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X', label='Cluster Centroids')
plt.title('K-Means Clustering and KNN Classification Distribution')
plt.xlabel('Normalized Feature 1')
plt.ylabel('Normalized Feature 2')
plt.legend()
plt.show()