import matplotlib.pyplot as plt
import os
from PIL import Image
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
    INPUT_SHAPE = (64, 64, 3)
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

def train_model(model,train_data,train_label):
    BATCH_SIZE = 4
    EPOCHS = 12
    METRICS = metrics=['accuracy',
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

if  __name__ == "__main__":
    folder = 'D:/Python/zero_to_hero_-scikit-learn-/CNN/Dataset'
    train_data, train_label = load_images_from_folder(folder)
    show_image(train_data, class_name, train_label)
    train_labels = to_categorical(train_label, len(class_name))  # Thay đổi dòng này
    train_data = np.asarray(train_data)
    model = create_model()
    train_model(model, train_data, train_labels)
