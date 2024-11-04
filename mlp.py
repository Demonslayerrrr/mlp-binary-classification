from sklearn.datasets import make_circles
import numpy as np

n_samples = 1000
np.random.seed(100)

x, y = make_circles(n_samples=n_samples, noise=0.03, random_state=100)

train_split = int(0.8 * len(x))
x_train, y_train = x[:train_split], y[:train_split]
x_test, y_test = x[train_split:], y[train_split:]


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

    def derivative_weights(self, y_true, y_pred, x):
        dL_dz = self.derivative(y_true, y_pred)
        return np.outer(dL_dz, x)

    def derivative_biases(self, y_true, y_pred):
        dL_dz = self.derivative(y_true, y_pred)
        return dL_dz

class BinaryClassificationModel:
    def __init__(self):
        self.hidden_layer_1 = Linear(2, 5)
        self.hidden_layer_2 = Linear(5, 5)
        self.output_layer = Linear(5, 1)

    def forward(self, x):
        hidden_1_output = self.hidden_layer_1.single_forward(x)
        relu_hidden_1_output = relu(hidden_1_output)
        
        hidden_2_output = self.hidden_layer_2.single_forward(relu_hidden_1_output)
        relu_hidden_2_output = relu(hidden_2_output)
        
        return self.output_layer.single_forward(relu_hidden_2_output)

model = BinaryClassificationModel()
bce = BCE()
y_logits = model.forward(x_train)
y_pred_probs = sigmoid(y_logits)

print("Logits output:", y_logits[:10])
print("Predicted probabilities:", y_pred_probs[:10])

print(f"Loss: {bce.loss_function(y_train,y_pred_probs)}")