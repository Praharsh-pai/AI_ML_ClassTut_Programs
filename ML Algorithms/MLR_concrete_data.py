import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.metrics import mean_squared_error, r2_score # type: ignore
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
file_path = 'concrete_data.csv'
concrete_data = pd.read_csv(file_path)
print("First few rows of the dataset:")
print(concrete_data.head())

X = concrete_data.iloc[:, :-1].values
y = concrete_data.iloc[:, 8].values

print('data X: \n',X)
print('data Y: \n',y)

labelencoder_X = LabelEncoder()
X[:, 7] = labelencoder_X.fit_transform(X[:, 7])

column_transformer = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), [7])
    ],
    remainder='passthrough'
)

X = column_transformer.fit_transform(X)
print('Catgorical Data: \n',X)

X = X.astype(int)
y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f'R-squared: {r2}')
print(f'Mean Squared Error: {mse}')
print(f'Training Score: {train_score}')
print(f'Test Score: {test_score}')
