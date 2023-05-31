import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from typing import Tuple

def uniformly_shuffles_copies(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p = np.random.permutation(a.shape[0])
    return a[p], b[p]

train_data = pd.read_csv('titanic/train.csv')
test_data = pd.read_csv('titanic/test.csv')
submission_data = pd.read_csv('titanic/gender_submission.csv')

# Удаление ненужных признаков
train_data = train_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
test_data = test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Обработка пропущенных значений
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

# Кодирование признаков
enc = OneHotEncoder()
train_data['Sex'] = enc.fit_transform(train_data['Sex'])
test_data['Sex'] = enc.transform(test_data['Sex'])
train_data['Embarked'] = enc.fit_transform(train_data['Embarked'])
test_data['Embarked'] = enc.transform(test_data['Embarked'])

# Разделение данных 
X_train = train_data.drop('Survived', axis=1)
y_train = train_data['Survived']
X_test = test_data

X_train, y_train = uniformly_shuffles_copies(X_train.values, y_train.values)

# Создание и обучение модели
knn = KNeighborsClassifier(
                            n_neighbors = 15,
                            algorithm = 'kd_tree', 
                            weights= 'distance', 
                            p= 1
                            # Параметры подобраны с помошью gridsearch 
                           )
knn.fit(X_train, y_train)

# Предсказание меток 
y_pred = knn.predict(X_test)
submission_labels = submission_data['Survived']

# Оценка точности предсказания
accuracy = accuracy_score(submission_labels, y_pred)
print("Точность предсказания:", accuracy)

# Вывод предсказанных меток
# print("Предсказанные метки:", y_pred)