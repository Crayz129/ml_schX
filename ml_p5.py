import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score

# Загрузка данных
train_data = pd.read_csv('titanic/train.csv')
test_data = pd.read_csv('titanic/test.csv')
submission_data = pd.read_csv('titanic/gender_submission.csv')

# Очистка данных и кодировка тексотвых признаков
train_data = train_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
test_data = test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
# Sex onehot encoding
oneh = OneHotEncoder(handle_unknown='ignore')
print(test_data['Sex'])
test_data['Sex'] = oneh.fit_transform(test_data[['Sex']]).toarray()
train_data['Sex'] = oneh.fit_transform(train_data[['Sex']]).toarray()

# Заполнение пропусков
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
test_data['Age'].fillna(train_data['Age'].median(), inplace=True)
test_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)

# Разделение на валидационную, тестовую и тренировочную выборки 
X_train = train_data.drop('Survived', axis=1)
y_train = train_data['Survived']
X_test = test_data

X_val = test_data
y_val = submission_data['Survived']

# Создание класса модели
class TitanicModel(nn.Module):
    def __init__(self, input_size):
        super(TitanicModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Создание модели
input_size = X_train.shape[1]
model = TitanicModel(input_size)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100
batch_size = 32

# Обучение модели
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    num_batches = len(X_train) // batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_X = torch.tensor(X_train[start_idx:end_idx].values, dtype=torch.float32)
        batch_y = torch.tensor(y_train[start_idx:end_idx].values, dtype=torch.float32).view(-1, 1)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    model.eval()
    val_X = torch.tensor(X_val.values.shape[0], dtype=torch.float32)
    val_y = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
    val_outputs = model(val_X)
    val_loss = criterion(val_outputs, val_y)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/num_batches:.4f}, Val Loss: {val_loss.item():.4f}")

# Оценка точности
test_X = torch.tensor(test_data.values, dtype=torch.float32)
test_outputs = model(test_X)
predictions = (test_outputs >= 0.5).view(-1).int().numpy()
accuracy = accuracy_score(y_val, predictions)
print("Точность предсказания:", accuracy)