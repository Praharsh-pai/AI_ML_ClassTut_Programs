import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore


file_path = 'car_evaluation.csv'
df = pd.read_csv(file_path)
df.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
label_encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le
X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
classification_rep = classification_report(y_test, y_pred, target_names=label_encoders['class'].classes_)
print(classification_rep)
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoders['class'].classes_, yticklabels=label_encoders['class'].classes_)
plt.title("Confusion Matrix for Random Forest Classifier")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()