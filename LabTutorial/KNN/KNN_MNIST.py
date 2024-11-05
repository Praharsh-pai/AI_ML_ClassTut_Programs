import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist # type: ignore
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
knn = KNeighborsClassifier()
param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
grid_search = GridSearchCV(knn, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_knn = grid_search.best_estimator_
print(f"Best KNN Parameters: {grid_search.best_params_}")
val_predictions_knn = best_knn.predict(X_val)
val_accuracy_knn = accuracy_score(y_val, val_predictions_knn)
print(f"Validation Accuracy with Optimized KNN: {val_accuracy_knn:.4f}")
test_predictions_knn = best_knn.predict(X_test)
test_accuracy_knn = accuracy_score(y_test, test_predictions_knn)
print(f"Test Accuracy with Optimized KNN: {test_accuracy_knn:.4f}")
kmeans = KMeans(n_clusters=10, random_state=42)
X_train_kmeans = kmeans.fit_predict(X_train)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
plt.figure(figsize=(10, 8))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=X_train_kmeans, cmap='viridis', s=1)
plt.title("K-Means Clustering of MNIST Digits (PCA-reduced to 2D)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar()
plt.show()