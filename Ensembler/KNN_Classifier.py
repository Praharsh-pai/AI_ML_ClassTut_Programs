import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

knn_model = KNeighborsClassifier(n_neighbors=5) 
knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)

print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Train Score: ', knn_model.score(X_train, y_train)*100)
print('Test Score: ', knn_model.score(X_test, y_test)*100)
