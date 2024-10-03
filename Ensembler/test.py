import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data_set = pd.read_csv('50_Startups.csv')
print("First few rows of the dataset:")
print(data_set.head())

labelencoder = LabelEncoder()
data_set['State'] = labelencoder.fit_transform(data_set['State'])

data_set['Profit_Class'] = pd.cut(data_set['Profit'], bins=3, labels=['Low', 'Medium', 'High'])

X = data_set.drop(['Profit', 'Profit_Class'], axis=1)
y = data_set['Profit_Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(),
    "Support Vector Machine": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

param_grids = {
    "Logistic Regression": {'C': [0.01, 0.1, 1, 10, 100]},
    "Support Vector Machine": {'kernel': ['linear', 'poly', 'rbf'], 'C': [0.1, 1, 10, 100]},
    "Decision Tree": {'max_depth': [3, 5, 10, None], 'min_samples_split': [2, 5, 10]},
    "K-Nearest Neighbors": {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
}

best_accuracies = {}

for model_name, model in models.items():
    print(f"Tuning {model_name}...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grids[model_name], cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Best Parameters for {model_name}: {best_params}")
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    best_accuracies[model_name] = accuracy
    print(f"{model_name} Accuracy after tuning: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

plt.figure(figsize=(10, 6))
plt.barh(list(best_accuracies.keys()), list(best_accuracies.values()), color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
plt.title('Tuned Model Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Models')
plt.xlim(0, 1)
plt.show()
