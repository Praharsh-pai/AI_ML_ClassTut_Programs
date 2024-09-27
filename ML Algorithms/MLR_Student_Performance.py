import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.metrics import mean_squared_error, r2_score # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore

file_path = 'Student_Performance.csv'
student_data = pd.read_csv(file_path)
print("First few rows of the dataset:")
print(student_data.head())

label_encoder = LabelEncoder()
student_data['Extracurricular Activities'] = label_encoder.fit_transform(student_data['Extracurricular Activities'])

X = student_data.drop(columns=['Performance Index (target dependent )'])  # Features
y = student_data['Performance Index (target dependent )']  # Target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)


train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f'Training Score: {train_score}')
print(f'Test Score: {test_score}')
