import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tkinter import Tk, filedialog
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from PIL import Image

device = 'cpu'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

model = torch.load('cv.pth')
model.eval()
image_path = 'path_to_image.jpg'
image = Image.open(image_path)
image_transformed = transform(image).unsqueeze(0)
image_transformed = image_transformed.to(device)
with torch.no_grad():
    output = model(image_transformed)

# Получите индекс класса с наибольшей вероятностью
predicted_class = torch.argmax(output).item()

# Определите класс на основе индекса
if predicted_class == 1:
    print("Кот")
else:
    print("Собака")