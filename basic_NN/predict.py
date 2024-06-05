import random
import math
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def loss(y_pred, y_true):
    return (y_pred - y_true) ** 2

def gradient_descent(x, y, learning_rate=0.01, num_iterations=1000):
    n = len(x[0])  # Number of features
    weights = [random.random() for _ in range(n + 1)]  # Initialize weights randomly
    for _ in range(num_iterations):
        for i in range(len(x)):
            y_pred = sigmoid(sum(weights[j] * x[i][j] for j in range(n)) + weights[-1])
            error = loss(y_pred, y[i])
            for j in range(n):
                weights[j] -= learning_rate * (y_pred - y[i]) * y_pred * (1 - y_pred) * x[i][j]
            weights[-1] -= learning_rate * (y_pred - y[i]) * y_pred * (1 - y_pred)  # Update bias
    return weights

# Example usage
x = [[1, 2], [2, 3], [3, 4]]  # Features
y = [0, 1, 0]  # Labels
weights = gradient_descent(x, y)
for i in range(len(x)):
    prediction = sigmoid(sum(x[i][j] * weights[j] for j in range(len(x[i]))) + weights[-1])
    print(f"Predicted probability for instance {i+1}: {prediction}")
