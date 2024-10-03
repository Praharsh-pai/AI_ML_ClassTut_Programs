import pandas as pd # type: ignore
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.svm import SVC # type: ignore

data_set = pd.read_csv('50_Startups.csv')
print("First few rows of the dataset:")
print(data_set.head())
# Separate features and target variable (Profit)
X = data_set.iloc[:, :-1].values
y = data_set.iloc[:, 4].values
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
column_transformer = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), [3])
    ],
    remainder='passthrough'
)
X = column_transformer.fit_transform(X)
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Convert 'Profit' into categories (e.g., Low, Medium, High)
y = pd.cut(y, bins=3, labels=['Low', 'Medium', 'High'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print('Train Score: ', classifier.score(X_train, y_train)*100)
print('Test Score: ', classifier.score(X_test, y_test)*100)