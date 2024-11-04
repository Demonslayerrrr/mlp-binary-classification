from sklearn.datasets import make_circles
import numpy as np

n_samples = 1000
np.random.seed(100)

x, y = make_circles(n_samples=n_samples, noise=0.03, random_state=100)

train_split = int(0.8 * len(x))
x_train, y_train = x[:train_split], y[:train_split]
x_test, y_test = x[train_split:], y[train_split:]

learning_rate = 0.01

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Linear:
    def __init__(self, input_features, output_features):
        self.weights = 0.1 * np.random.randn(input_features, output_features)
        self.biases = np.zeros(output_features)

    def single_forward(self, x):
        return np.dot(x, self.weights) + self.biases

class BCE:
    def __init__(self):
        pass

    def loss_function(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def derivative(self, y_true, y_pred):
        return y_pred - y_true

    def derivative_weights(self, y_true, y_pred, previous_layer_output):
        dL_dz = self.derivative(y_true, y_pred)
        return np.dot(previous_layer_output.T, dL_dz) / len(y_true)

    def derivative_biases(self, y_true, y_pred):
        dL_dz = self.derivative(y_true, y_pred)
        return np.mean(dL_dz, axis=0)

class BinaryClassificationModel:
    def __init__(self):
        self.hidden_layer_1 = Linear(2, 5)
        self.hidden_layer_2 = Linear(5, 5)
        self.output_layer = Linear(5, 1)

    def forward(self, x):
        self.hidden_1_output = self.hidden_layer_1.single_forward(x)
        self.relu_hidden_1_output = relu(self.hidden_1_output)
        
        self.hidden_2_output = self.hidden_layer_2.single_forward(self.relu_hidden_1_output)
        self.relu_hidden_2_output = relu(self.hidden_2_output)
        
        self.output_logits = self.output_layer.single_forward(self.relu_hidden_2_output)
        return self.output_logits

    def backward(self, y_true, y_pred):
        bce = BCE()

        self.output_layer_error = bce.derivative(y_true, y_pred).reshape(-1, 1)

        self.d_output_layer_weights = bce.derivative_weights(y_true, y_pred, self.relu_hidden_2_output)
        self.d_output_layer_biases = bce.derivative_biases(y_true, y_pred)

        self.hidden_layer_2_error = np.dot(self.output_layer_error, self.output_layer.weights.T) * (self.relu_hidden_2_output > 0)

        self.d_hidden_2_weights = bce.derivative_weights(self.hidden_layer_2_error, self.relu_hidden_1_output)
        self.d_hidden_2_biases = np.mean(self.hidden_layer_2_error, axis=0)

        self.hidden_layer_1_error = np.dot(self.hidden_layer_2_error, self.hidden_layer_2.weights.T) * (self.relu_hidden_1_output > 0)

        self.d_hidden_1_weights = bce.derivative_weights(self.hidden_layer_1_error, self.hidden_1_output)
        self.d_hidden_1_biases = np.mean(self.hidden_layer_1_error, axis=0)

        return (self.d_output_layer_weights, self.d_output_layer_biases,
                self.d_hidden_2_weights, self.d_hidden_2_biases,
                self.d_hidden_1_weights, self.d_hidden_1_biases)

    def step(self, d_output_weights, d_output_biases,
             d_hidden_2_weights, d_hidden_2_biases,
             d_hidden_1_weights, d_hidden_1_biases):
        
        self.output_layer.weights -= learning_rate * d_output_weights
        self.output_layer.biases -= learning_rate * d_output_biases

        self.hidden_layer_2.weights -= learning_rate * d_hidden_2_weights
        self.hidden_layer_2.biases -= learning_rate * d_hidden_2_biases

        self.hidden_layer_1.weights -= learning_rate * d_hidden_1_weights
        self.hidden_layer_1.biases -= learning_rate * d_hidden_1_biases

    def fit(self, epochs):
        bce = BCE()

        for epoch in range(epochs):
            y_logits = self.forward(x_train)
            y_pred_probs = sigmoid(y_logits)
            loss = bce.loss_function(y_train, y_pred_probs)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

            self.step(*self.backward(y_train, y_pred_probs))

model = BinaryClassificationModel()
model.fit(epochs=1000)
