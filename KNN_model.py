##first model, i want to create model kn neighbors  classifier in compute binary  classification.
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
## i have a file dataset and then i will split  it into train and test set
data = pd.read_csv("D:/Python/zero_to_hero_-scikit-learn-/fake_data.csv",header=0)
X = data[['day', 'total_day']].values
y = data[["class"]].values
X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5, stratify=y)
test_pre = {}
train_pre = {}
neighbors = range(1,11)
for i in range (1,11):
    knn = KNeighborsClassifier(n_neighbors=i) #create object of kNN Classifier with n_neighbor=3
    #fit the training data on the kNN Model
    knn.fit(X_train, y_train)
    test_pre[i] =knn.score(X_test, y_test)#predict the response for test dataset
    train_pre[i] =knn.score(X_train, y_train)
    print("{}".format(i))

## init plot
plt.title("observe key affect to predict in model")
plt.xlabel(r"Number of Neighbour $k$")
plt.ylabel("Accuracy Score")
plt.plot(neighbors,test_pre.values(),label="test-score")
plt.plot(neighbors,train_pre.values(),label="train-score")
plt.show()