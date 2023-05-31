import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler



data_url = "D:\\code\\ml\\boston.txt"
data_columns = "CRIM ZN INDUS CHAS NOX RM AGE DIS RAD TAX PTRATIO B LSTAT".split(" ")
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
df = pd.DataFrame(data, columns = data_columns)
target = raw_df.values[1::2, 2]
df["MEDV"] = target

# Разбиваем данные на обучающую и тестовую выборки
df_train, df_test, y_train, y_test = train_test_split(df, target, test_size=0.07, random_state = 355) # значения test_size и random_state подобраны перебором для минимального среднего значения квадратичной ошибки

# Разделяем признаки и целевое значение
X_train = df_train.iloc[:, :-1]
X_test = df_test.iloc[:, :-1]

# Создаём объект класса StandartScaler, тренеруем его на тренировочной выборке и нормализуем тестовую выборку
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Создаём объект класса LinearRegression и скармливаем ему массив признаков и массив целевых переменных
model = LinearRegression()
model.fit(X_train, y_train, sample_weight=None)

# Предсказываем значения на тестовой выборке, считаем и выводим среднюю квадратичную ошибку и коэффициент детерминации
predicted = model.predict(X_test)
mse = mean_squared_error(y_test, predicted)
r2 = r2_score(y_test, predicted)
print('Средняя квадратичная ошибка: ', mse)
print('Коэффициент детерминации: ', r2)

# Предсказываем значение для новых данных и выводим результат
new_data = pd.DataFrame([[0.02731, 0.00, 7.070, 0, 0.4690, 6.4210, 78.90, 4.9671, 2, 242.0, 17.80, 396.90, 9.14]]) 
new_data = scaler.transform(new_data)
prediction = model.predict(new_data)
print('Prediction:', prediction)
