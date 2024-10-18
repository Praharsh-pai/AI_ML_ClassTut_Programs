import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.neural_network import MLPClassifier # type: ignore
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # type: ignore
from tensorflow.keras.datasets import mnist # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=10, batch_size=128, random_state=42)
mlp.fit(X_train, y_train)
val_predictions = mlp.predict(X_val)
val_accuracy = accuracy_score(y_val, val_predictions)
print(f"Validation Accuracy: {val_accuracy:.4f}")
test_predictions = mlp.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f"Test Accuracy: {test_accuracy:.4f}")
n_samples = 5
sample_indices = np.random.randint(0, X_test.shape[0], n_samples)
plt.figure(figsize=(15, 3))
for i, idx in enumerate(sample_indices):
    plt.subplot(1, n_samples, i+1)
    plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f"True: {y_test[idx]}, Pred: {test_predictions[idx]}")
    plt.axis('off')
plt.show()

def predict_user_image(mlp, img_path):
    img = image.load_img(img_path, target_size=(28, 28), color_mode="grayscale")
    img = image.img_to_array(img)
    img = img.reshape(1, 28*28) / 255.0
    prediction = mlp.predict(img)
    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.title(f"Predicted Digit: {prediction[0]}")
    plt.axis('off')
    plt.show()
    return prediction[0]

img_path = './images/7b.jpg'
predicted_digit = predict_user_image(mlp, img_path)
print(f"The MLP model predicted: {predicted_digit}")
