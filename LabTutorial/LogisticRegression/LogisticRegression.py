import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import LabelEncoder, StandardScaler # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.metrics import classification_report, accuracy_score # type: ignore

file_path = 'train.csv'
df = pd.read_csv(file_path)
print(df.head())

columns_to_drop = ['Name', 'Ticket', 'Cabin']
df = df.drop(columns=columns_to_drop, errors='ignore')
df = df.assign(Age=df['Age'].fillna(df['Age'].median()), 
               Embarked=df['Embarked'].fillna(df['Embarked'].mode()[0]))

label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

if 'Survived' in df.columns:
    X = df.drop('Survived', axis=1)
    y = df['Survived']
else:
    print("No target column 'Survived' found in the dataset.")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

log_reg_model = LogisticRegression(max_iter=200)
log_reg_model.fit(X_train, y_train)

y_pred = log_reg_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.4f}')

classification_rep = classification_report(y_test, y_pred)
print(classification_rep)
