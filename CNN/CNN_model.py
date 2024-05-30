import torch
from torch.nn import Module
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import os
from PIL import Image
from torchvision import transforms

class_name = ('dog', 'cat')

# Define the model architecture
class Model(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 64x64
        self.conv1 = nn.Conv2d(3, 6, 3)  # 62x62
        self.max1 = nn.MaxPool2d(2, 2)  # 31x31

        self.conv2 = nn.Conv2d(6, 16, 3)  # 29x29
        self.max2 = nn.MaxPool2d(2, 2)  # 14x14

        self.fc1 = nn.Linear(16 * 14 * 14, 240)
        self.fc2 = nn.Linear(240, 120)
        self.fc3 = nn.Linear(120, 80)
        self.fc4 = nn.Linear(80, 2)  # Assuming 2 classes (dog, cat)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.max1(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.max2(x)
        x = x.view(-1, 16 * 14 * 14)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def train(model, data, epochs=10, lr=0.01, weight_decay=3e-5):
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion.to(device)
    
    for epoch in range(epochs):
        model.train()
        for inputs, labels in data:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch+1}/{epochs}, Loss: {loss.item()}")

def solution(paths):
    file_image = []
    for path in paths:
        for file in os.listdir(path):
            if file.endswith(".jpg"):
                file_image.append((os.path.join(path, file), os.path.basename(path)))
    return file_image

def solution_data(file_image):
    resize = []
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    for file, label in file_image:
        image = Image.open(file)
        image = transform(image)
        label = class_name.index(label)  # Convert label to index
        resize.append((image, label))
    
    images, labels = zip(*resize)  # Unzip into separate lists of images and labels
    images = torch.stack(images)  # Stack images into a single tensor
    labels = torch.tensor(labels)  # Convert labels to a single tensor
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=32, shuffle=True)

if __name__ == '__main__':
    paths = ['/CNN/Dataset/dog', '/CNN/Dataset/cat']  # Thay bằng đường dẫn thực tế của bạn
    file_image = solution(paths)
    data_loader = solution_data(file_image)
    
    model = Model()
    train(model, data_loader)
