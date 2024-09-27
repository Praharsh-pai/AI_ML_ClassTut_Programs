import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import pandas as pd # type: ignore
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore

data_set = pd.read_csv('50_Startups.csv')
print("First few rows of the dataset:")
print(data_set.head())

X = data_set.iloc[:, :-1].values
y = data_set.iloc[:, 4].values

# print('data X: \n',X)
# print('data Y: \n',y)

labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

column_transformer = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), [3])
    ],
    remainder='passthrough'
)

X = column_transformer.fit_transform(X)
# print('Catgorical Data: \n',X)

X = X.astype(int)
y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print('Train Score: ', regressor.score(X_train, y_train)*100)
print('Test Score: ', regressor.score(X_test, y_test)*100)
