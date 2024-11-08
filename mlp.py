import numpy as np
from sklearn.datasets import make_circles
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

n_samples = 1000
np.random.seed(100)

x, y = make_circles(n_samples=n_samples, noise=0.03, random_state=100)
train_split = int(0.8 * len(x))
x_train, y_train = x[:train_split], y[:train_split]
x_test, y_test = x[train_split:], y[train_split:]

class Relu:
    def relu(self, x):
        self.x = x
        return np.maximum(0, x)

    def derivative(self, derivative_value):
        return derivative_value * (self.x > 0)

def sigmoid(x):
    x = np.clip(x, -709, 709)
    return 1 / (1 + np.exp(-x))

class Linear:
    def __init__(self, input_features, output_features):
        limit = np.sqrt(2 / input_features)
        self.weights = np.random.randn(input_features, output_features) * limit
        self.biases = np.zeros((1, output_features))

    def single_forward(self, x):
        self.x = x
        return np.dot(self.x, self.weights) + self.biases

    def derivatives(self, derivative_values):
        n = self.x.shape[0]
        self.derivative_weights = np.dot(self.x.T, derivative_values) / n
        self.derivative_biases = np.sum(derivative_values, axis=0, keepdims=True) / n
        self.derivative_layer = np.dot(derivative_values, self.weights.T)

class BCE:
    def loss_function(self, y_true, y_pred):
        y_true = y_true.reshape(-1, 1)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

class GradientDescent:
    def __init__(self, lr=0.1):
        self.learning_rate = lr

    def update_values(self, layer):
        layer.weights -= self.learning_rate * layer.derivative_weights
        layer.biases -= self.learning_rate * layer.derivative_biases

class BinaryClassificationModel:
    def __init__(self):
        self.input_layer = Linear(2, 5)
        self.hidden_layer_1 = Linear(5, 5)
        self.hidden_layer_2 = Linear(5, 5)
        self.output_layer = Linear(5, 1)

        self.relu1 = Relu()
        self.relu2 = Relu()
        self.relu3 = Relu()

    def forward(self, x):
        input_output = self.input_layer.single_forward(x)
        relu_input_output = self.relu1.relu(input_output)
        hidden_1_output = self.hidden_layer_1.single_forward(relu_input_output)

        relu_hidden_1_output = self.relu2.relu(hidden_1_output)
        hidden_2_output = self.hidden_layer_2.single_forward(relu_hidden_1_output)
        relu_hidden_2_output = self.relu3.relu(hidden_2_output)

        output = self.output_layer.single_forward(relu_hidden_2_output)
        return output

    def backward(self, delta):
        self.output_layer.derivatives(delta)
        derivative_layer = self.output_layer.derivative_layer
        derivative_layer = self.relu3.derivative(derivative_layer)

        self.hidden_layer_2.derivatives(derivative_layer)
        derivative_layer = self.hidden_layer_2.derivative_layer
        derivative_layer = self.relu2.derivative(derivative_layer)

        self.hidden_layer_1.derivatives(derivative_layer)
        derivative_layer = self.hidden_layer_1.derivative_layer
        derivative_layer = self.relu1.derivative(derivative_layer)

        self.input_layer.derivatives(derivative_layer)

    def fit(self, epochs):
        bce = BCE()
        gradient_descent = GradientDescent(lr=0.1)

        loss_values = []

        epochs_values = []

        for epoch in tqdm(range(epochs)):
            y_logits = self.forward(x_train)
            y_pred_probs = sigmoid(y_logits)

            loss = bce.loss_function(y_train, y_pred_probs)
            delta = y_pred_probs - y_train.reshape(-1, 1)

            self.backward(delta)

            gradient_descent.update_values(self.output_layer)
            gradient_descent.update_values(self.hidden_layer_2)
            gradient_descent.update_values(self.hidden_layer_1)
            gradient_descent.update_values(self.input_layer)

            if epoch % 100 == 0:
                print(f'Epoch: {epoch} | Loss: {loss:.4f}')

                epochs_values.append(epoch)

                loss_values.append(loss)
        return loss_values,epochs

    def predict(self, x):
        y_logits = self.forward(x)
        y_pred_probs = sigmoid(y_logits)
        return (y_pred_probs > 0.5).astype(int)

model = BinaryClassificationModel()
loss_vals,epochs_vals=model.fit(epochs=5000)

y_pred = model.predict(x_test)
accuracy = np.mean(y_pred.flatten() == y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')




def plot_decision_boundary(model, x, y, resolution=0.01):
    x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
    y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5


    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution))
    grid_points = np.c_[xx.ravel(), yy.ravel()]


    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)


    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')  
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolor='k', cmap='coolwarm', s=20) 
    plt.title("Decision Boundary of the Model")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


plot_decision_boundary(model, x, y)
