import matplotlib as plt

# first we need create a function to read imgae and labels
class_name = ['dog', 'cat']
def show_image(train_images,
               class_name,
               train_labels,
               nb_samples = 12):
    plt.figure(figure_size =(12,12)) # create images show with array [12,12]
    for i in nb_samples:
        plt.subplot(3, 4, i + 1)  # set the number of rows and columns. The figure have  3*4=12 subplots
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.imshow(class_name[train_labels[i][0]])
    plt.show()
