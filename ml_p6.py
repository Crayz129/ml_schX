import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tkinter import Tk, filedialog
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from PIL import Image

lr = 0.001
batch_size = 100
epochs = 10

device="cpu"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

# Load train and test data
train_dir = r'D:\code\ml\PetImages'
train_list = []

for path, currentDirectory, files in os.walk(train_dir):
    for file in files:
        train_list.append(os.path.join(path, file))

class dataset(torch.utils.data.Dataset):
    def __init__(self,file_list,transform = None):
        self.file_list=file_list
        self.transform=transform

    def __len__(self):
        self.filelength =len(self.file_list)
        return self.filelength

    def __getitem__(self,idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = os.path.split(os.path.split(img_path)[-2])[-1]
        if label == 'Dog':
            label = torch.tensor(1)
        elif label == 'Cat':
            label = torch.tensor(0)

        return img_transformed,label

train_list, val_list = train_test_split(train_list, test_size = 0.2)

train_data = dataset(train_list,transform=transform)
val_data = dataset(val_list,transform=transform)

train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = batch_size,shuffle = True)
val_loader = torch.utils.data.DataLoader(dataset = val_data, batch_size = batch_size,shuffle = True)

class Cnn(nn.Module):
    def __init__(self):
        super(Cnn,self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3,padding=0,stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=3,padding=0,stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,padding=0,stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(3*3*64,10)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10,2) 
        self.relu = nn.ReLU() # sigmoid

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0),-1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

model = Cnn().to(device)
model.train()
optimizer = optim.Adam(params = model.parameters(),lr =0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    for data, label in train_loader:
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output,label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = ((output.argmax(dim=1)==label).float().mean())
        epoch_accuracy += acc/len(train_loader)
        epoch_loss += loss/len(train_loader)

    print('Epoch : {}, train accuracy : {}, train loss : {}'.format(epoch+1, epoch_accuracy,epoch_loss))

    with torch.no_grad():
        epoch_val_accuracy =0
        epoch_val_loss = 0
        for data,label in  val_loader:
            data= data.to(device)

            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(output,label)

            acc = ((output.argmax(dim=1)==label).float().mean())
            epoch_val_accuracy += acc/len(val_loader)
            epoch_val_loss += val_loss/len(val_loader)
        print('Epoch : {}, val_accuracy : {}, val_loss : {}'.format(epoch+1, epoch_val_accuracy,epoch_val_loss))

test_dir = r'D:\code\ml\TestImages'  # Directory containing test images
test_list = []

for path, currentDirectory, files in os.walk(test_dir):
    for file in files:
        test_list.append(os.path.join(path, file))

test_data = dataset(test_list, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)

model.eval()

correct = 0
total = 0

with torch.no_grad():
    for data, labels in test_loader:
        data = data.to(device)
        labels = labels.to(device)

        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print('Test Accuracy: {:.2f}%'.format(accuracy))

if accuracy >= 80:
    model_path = 'cv.pth'
    torch.save(model, model_path)
else:
    print('wtf lol')

''' 
    начать работать с rgb
    предобученные модели vgg resnet -> вытащить низкоклассовые фильтры
'''