
import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
from sklearn.metrics import r2_score # type: ignore

file_path = 'Student_Performance.csv'
student_data = pd.read_csv(file_path)
print("First few rows of the dataset:")
print(student_data.head())

labelencoder = LabelEncoder()
student_data['Extracurricular Activities'] = labelencoder.fit_transform(student_data['Extracurricular Activities'])

X = student_data.drop(columns=['Performance Index (target dependent )']) # Features
y = student_data['Performance Index (target dependent )'] # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = RandomForestRegressor(n_estimators=1000, random_state=42)
regressor.fit(X_train, y_train)

y_train_pred = regressor.predict(X_train)
y_test_pred = regressor.predict(X_test)


train_score = r2_score(y_train, y_train_pred)
test_score = r2_score(y_test, y_test_pred)

print(f"Training R-squared score: {train_score}")
print(f"Test R-squared score: {test_score}")
