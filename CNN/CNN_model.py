import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

class_name = ['cat', 'dog']
# create func to read image from folder
def load_images_from_folder(folder):
    images = []
    labels = []
    train_labels = []
    i = 0
    j = 0
    for class_names in os.listdir(folder):
        class_folder = os.path.join(folder, class_names)
        if os.path.isdir(class_folder):
            for filename in os.listdir(class_folder):
                train_labels.append([])
                img_path = os.path.join(class_folder, filename)
                if img_path.endswith(".jpg") or img_path.endswith(".png"):
                    img = Image.open(img_path)
                    img = img.resize((64, 64))  # Thay đổi kích thước hình ảnh nếu cần
                    img_array = np.array(img) / 255.0  # Chuẩn hóa 
                    images.append(img_array)
                    train_labels[i].append(j)
                    i += 1
            j += 1

    show_image (images,class_name,train_labels)

# first we need create a function to read imgae and labels

def show_image(train_images,
               class_name,
               train_labels,
               nb_samples = 12):
    plt.figure(figsize =(12,12)) # create images show with array [12,12]
    for i in range(0,nb_samples):
        plt.subplot(3, 4, i + 1)  # set the number of rows and columns. The figure have  3*4=12 subplots
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        if train_labels[i]:  # Kiểm tra xem train_labels[i] có phần tử hay không trước khi truy cập
            plt.xlabel(class_name[train_labels[i][0]])
    plt.show()

if  __name__ == "__main__":
    folder = 'D:/Python/zero_to_hero_-scikit-learn-/CNN/Dataset'
    load_images_from_folder(folder)