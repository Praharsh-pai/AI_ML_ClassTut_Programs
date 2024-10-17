import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

data = pd.read_csv('spam_or_not_spam.csv')

print("\nCheck for missing values:")
print(data.isnull().sum())
data = data.dropna()
print("\nBasic statistics of the dataset:")
print(data.describe())

vectorizer = TfidfVectorizer(max_features=5000)
features = vectorizer.fit_transform(data['email']).toarray()
target = data['label']  

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
scaler = StandardScaler(with_mean=False) 
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'\nAccuracy: {accuracy:.4f}')
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
print("\nFeature ranking:")
for f in range(X_train.shape[1]):
    print(f"{f + 1}. Feature {indices[f]} ({importances[indices[f]]:.4f})")
plt.figure()
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()

def predict_spam(sentence):
    sentence_transformed = vectorizer.transform([sentence]).toarray()
    sentence_transformed = scaler.transform(sentence_transformed)
    prediction = model.predict(sentence_transformed)
    return "Spam" if prediction == 1 else "Not Spam"

# new_sentence = "Congratulations! You've won a free ticket."
new_sentence = "Hey, are we still meeting for lunch tomorrow?"
print(f"\nThe sentence '{new_sentence}' is: {predict_spam(new_sentence)}")




