

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Load Dataset
df = pd.read_csv('model/Titanic-Dataset.csv')

# 2. Explore Dataset Structure
print('Shape:', df.shape)
print('Columns:', df.columns.tolist())
df.info()
print(df.sample(5))

# 3. Preprocess Data
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Fare']
target = 'Survived'

# Handle missing values
for col in ['Age', 'Fare']:
    df[col] = df[col].fillna(df[col].median())
df['Sex'] = df['Sex'].fillna(df['Sex'].mode()[0])

# Encode categorical variables
le_sex = LabelEncoder()
df['Sex'] = le_sex.fit_transform(df['Sex'])

# Feature scaling
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

X = df[features]
y = df[target]

# 4. Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 5. Evaluate Model
y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 6. Save Model
joblib.dump(model, 'model/titanic_survival_model.pkl')
print('Model saved as model/titanic_survival_model.pkl')

# 7. Reload and Predict
test_sample = X_test.iloc[0:1]
loaded_model = joblib.load('model/titanic_survival_model.pkl')
pred = loaded_model.predict(test_sample)
print('Sample input:', test_sample.values)
print('Predicted survival:', 'Survived' if pred[0] == 1 else 'Did Not Survive')
