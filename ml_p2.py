import numpy as np
from typing import Tuple
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Функция для согласованного перемешивания данных и лэйблов
def uniformly_shuffles_copies(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p = np.random.permutation(a.shape[0])
    return a[p], b[p]

# Загрузка датасета 
iris = load_iris()

# Разделение данных
X = iris.data
y = iris.target

X, y = uniformly_shuffles_copies(X, y)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=52)

# Создание и обучение модели SVM
model = svm.SVC()
model.fit(X_train, y_train)

# Предсказание значений для тестовой выборки
y_pred = model.predict(X_test)

# Оценка точности модели
accuracy = accuracy_score(y_test, y_pred)
print("Точность модели SVM:", accuracy)

# Пример предсказания для новых данных
new_data = [[5.1,3.5,1.4,0.2]]
predicted_iris = model.predict(new_data)
print("Предсказанный класс для новых данных:", predicted_iris)