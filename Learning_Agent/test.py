import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

file_path = 'spam_or_not_spam.csv'
data = pd.read_csv(file_path)

print(data.head())
print(data.info())

data = data.dropna()

X = data['email']  
y = data['label']  

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, 'decision_tree_text_model.pkl')

new_email = ["hello how r u? is  everything ok? received ur mail regarding issues in hostel"]
new_email_transformed = vectorizer.transform(new_email)
prediction = model.predict(new_email_transformed)

print(f"Prediction for the new email: {prediction}")
