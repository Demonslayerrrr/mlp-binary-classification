from sklearn.datasets import make_circles
import numpy as np

n_samples = 1000
np.random.seed(100)

x, y = make_circles(n_samples=n_samples, noise=0.03, random_state=100)

train_split = int(0.8 * len(x))
x_train, y_train = x[:train_split], y[:train_split]
x_test, y_test = x[train_split:], y[train_split:]


class Relu:
    def relu(self,x):
        self.x = x
        return np.maximum(0,self.x)
    def derivative(self,derivative_value):
        self.x = derivative_value.copy()
        self.x[self.x <= 0] = 0
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Linear:
    def __init__(self, input_features, output_features):
        self.weights = 0.1 * np.random.randn(input_features, output_features)
        self.biases = np.zeros((1,output_features))


    def single_forward(self, x):
        self.x =x
        return np.dot(self.x, self.weights) + self.biases
    
    def derivatives(self,derivative_values):
        self.derivative_weights = np.dot(np.array(self.x).T, derivative_values)
        self.derivative_biases = np.sum(derivative_values, axis=0, keepdims=True)

        self.derivative_layer = np.dot(derivative_values,self.weights.T)

class BCE:
    def loss_function(self, y_true:np.ndarray, y_pred:np.ndarray):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def derivative(self, y_true:np.ndarray, y_pred:np.ndarray):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        gradient = - (y_true / y_pred) + ((1 - y_true) / (1 - y_pred))
        return gradient

class GradientDescent:
    def __init__(self,lr=0.01):
        self.learning_rate = lr
    
    def update_values(self,layer:Linear):
        layer.weights -= self.learning_rate*layer.derivative_weights
        layer.biases -= self.learning_rate*layer.derivative_biases

class BinaryClassificationModel:
    def __init__(self):
        self.input_layer = Linear(2,5)
        self.hidden_layer_1 = Linear(5, 5)
        self.hidden_layer_2 = Linear(5, 5)
        self.output_layer = Linear(5, 1)

        self.relu = Relu()

    def forward(self, x):
        # Pass through the input layer first
        input_output = self.input_layer.single_forward(x)
        relu_input_output = self.relu.relu(input_output)
        
        # Then pass through hidden layers
        hidden_1_output = self.hidden_layer_1.single_forward(relu_input_output)
        relu_hidden_1_output = self.relu.relu(hidden_1_output)
        
        hidden_2_output = self.hidden_layer_2.single_forward(relu_hidden_1_output)
        relu_hidden_2_output = self.relu.relu(hidden_2_output)
        
        # Finally, pass through the output layer
        return self.output_layer.single_forward(relu_hidden_2_output)
    def backward(self, loss_deriv):
        self.output_layer.derivatives(loss_deriv)
        self.relu.derivative(self.output_layer.derivative_layer)

        self.hidden_layer_1.derivatives(self.relu.x)
        self.relu.derivative(self.hidden_layer_2.derivative_layer)

        self.hidden_layer_2.derivatives(self.relu.x)
        self.relu.derivative(self.hidden_layer_1.derivative_layer)

        self.input_layer.derivatives(self.relu.x)

    def fit(self, epochs):
        bce = BCE()
        gradient_descent = GradientDescent()

        for epoch in range(epochs):

            y_logits = self.forward(x_train)
            y_pred_probs = sigmoid(y_logits)


            loss = bce.loss_function(y_train, y_pred_probs)


            loss_deriv = bce.derivative(y_train, y_pred_probs)


            self.backward(loss_deriv)


            gradient_descent.update_values(self.output_layer)
            gradient_descent.update_values(self.hidden_layer_2)
            gradient_descent.update_values(self.hidden_layer_1)
            gradient_descent.update_values(self.input_layer)

            if epoch % 100 == 0:
                print(f'Epoch: {epoch} | Loss: {loss}')


    



model = BinaryClassificationModel()

model.fit(epochs=1000)