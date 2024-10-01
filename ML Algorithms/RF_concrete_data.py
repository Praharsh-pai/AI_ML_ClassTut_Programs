
import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
# from sklearn.preprocessing import LabelEncoder # type: ignore
from sklearn.metrics import r2_score # type: ignore

file_path = 'concrete_data.csv'
data = pd.read_csv(file_path)
print("First few rows of the dataset:")
print(data.head())

# labelencoder = LabelEncoder()
# data['State'] = labelencoder.fit_transform(data['State'])

X = data.drop(columns=['Strength']) # Features
y = data['Strength'] # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

y_train_pred = regressor.predict(X_train)
y_test_pred = regressor.predict(X_test)


train_score = r2_score(y_train, y_train_pred)
test_score = r2_score(y_test, y_test_pred)

print(f"Training R-squared score: {train_score*100}")
print(f"Test R-squared score: {test_score*100}")
