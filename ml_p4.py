import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from typing import Tuple

def uniformly_shuffles_copies(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p = np.random.permutation(a.shape[0])
    return a[p], b[p]

train_data = pd.read_csv('titanic/train.csv')
test_data = pd.read_csv('titanic/test.csv')
submission_data = pd.read_csv('titanic/gender_submission.csv')

# Удаление ненужных столбцов
train_data = train_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
test_data = test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Кодирование признаков
label_encoder = LabelEncoder()
train_data['Sex'] = label_encoder.fit_transform(train_data['Sex'])
test_data['Sex'] = label_encoder.transform(test_data['Sex'])
train_data['Embarked'] = label_encoder.fit_transform(train_data['Embarked'])
test_data['Embarked'] = label_encoder.transform(test_data['Embarked'])

# Замена пропущенных значений средним значением
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
test_data['Age'].fillna(train_data['Age'].median(), inplace=True)
test_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

# Разделение данных
X_train = train_data.drop('Survived', axis=1)
y_train = train_data['Survived']
X_test = test_data

X_train, y_train = uniformly_shuffles_copies(X_train.values, y_train.values)

# Создание и обучение модели 
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Предсказание выживания на тестовых данных
y_pred = model.predict(X_test)

submission_labels = submission_data['Survived']

# Оценка точности предсказания
accuracy = accuracy_score(submission_labels, y_pred)
print("Точность предсказания:", accuracy)
# print(y_pred)