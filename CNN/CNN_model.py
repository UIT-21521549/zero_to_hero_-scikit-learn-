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
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.metrics import Precision, Recall
class_name = ['cat', 'dog']
# create func to read image from folder
def load_images_from_folder(folder):
    images = []
    train_labels = []
    i = 0
    for class_names in os.listdir(folder):
        class_folder = os.path.join(folder, class_names)
        if os.path.isdir(class_folder):
            for filename in os.listdir(class_folder):
                img_path = os.path.join(class_folder, filename)
                if img_path.endswith(".jpg") or img_path.endswith(".png"):
                    img = Image.open(img_path)
                    img = img.resize((64, 64))  # Thay đổi kích thước hình ảnh nếu cần
                    img_array = np.array(img) / 255.0  # Chuẩn hóa 
                    images.append(img_array)
                    train_labels.append(i)  # Thay đổi dòng này
            i += 1

    return images, train_labels


# first we need create a function to read imgae and labels

def show_image(train_images,
               class_name,
               train_labels,
               nb_samples = 12):
    plt.figure(figsize =(12,12)) # create images show with 12x12 inchs
    for i in range(0,nb_samples):
        plt.subplot(3, 4, i + 1)  # set the number of rows and columns. The figure have  3*4=12 subplots
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        if train_labels[i]:  # Kiểm tra xem train_labels[i] có phần tử hay không trước khi truy cập
            plt.xlabel(class_name[train_labels[i][0]])
    plt.show()
    train_labels = to_categorical(train_labels, len(class_name))

def create_model():
    INPUT_SHAPE = (64, 64, 4)
    FILTER1_SIZE = 32
    FILTER2_SIZE = 64
    FILTER_SHAPE = (3, 3)
    POOL_SHAPE = (2, 2)
    FULLY_CONNECT_NUM = 128
    NUM_CLASSES = len(class_name)

    # Model architecture implementation
    model = Sequential()
    model.add(Conv2D(FILTER1_SIZE, FILTER_SHAPE, activation='relu', input_shape=INPUT_SHAPE))
    model.add(MaxPooling2D(POOL_SHAPE))
    model.add(Conv2D(FILTER2_SIZE, FILTER_SHAPE, activation='relu'))
    model.add(MaxPooling2D(POOL_SHAPE))
    model.add(Flatten())
    model.add(Dense(FULLY_CONNECT_NUM, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model

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
def train_model(model,train_data,train_label):
    BATCH_SIZE = 4
    EPOCHS = 12
    METRICS =['accuracy',
                        Precision(name='precision'),
                        Recall(name='recall')]
    
    # Compile the model before

    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics = METRICS)

    # Train the model
    training_history = model.fit(train_data, train_label,
                        epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_data=(train_data, train_label))

if __name__ == '__main__':
    paths = ['/CNN/Dataset/dog', '/CNN/Dataset/cat']  # Thay bằng đường dẫn thực tế của bạn
    file_image = solution(paths)
    data_loader = solution_data(file_image)
    
    model = Model()
    train(model, data_loader)
if  __name__ == "__main__":
    folder = 'D:/Python/zero_to_hero_-scikit-learn-/CNN/Dataset'
    train_data, train_label = load_images_from_folder(folder)
    train_labels = to_categorical(train_label, len(class_name))  # Thay đổi dòng này
    train_data = np.asarray(train_data)
    print(train_data.shape)
    model = create_model()
    train_model(model, train_data, train_labels)
