from sklearn.datasets import make_circles
import numpy as np





n_samples = 1000

epochs = 1000

learning_rate = 0.01

random_seed = np.random.seed(100)
np.random.seed(100)

x,y = make_circles(n_samples=n_samples,
                       noise=0.03,
                       random_state=random_seed)



# plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.RdYlBu)
# plt.show()

x, y = make_circles(n_samples=n_samples, noise=0.03, random_state=100)

train_split = int(0.8 * len(x))
x_train, y_train = x[:train_split], y[:train_split]
x_test, y_test = x[train_split:], y[train_split:]


def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def accuracy_fn(y_true,y_pred):
  correct = np.equal(y_true,y_pred).sum().item()
  acc = (correct)/len(y_pred)*100

  return acc

class Relu:
    def relu(self,x):
        self.x = x
        return np.maximum(0,self.x)
    def derivative(self,derivative_value):
        self.x = derivative_value.copy()
        self.x[self.x <= 0] = 0
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def linear(x,w,b):
    a = np.dot(x,w)+b
    return relu(a),a

def bce(y_true,y_pred, epsilon=1e-15):

    y_pred = np.clip(y_pred,epsilon,1-epsilon)

    bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return bce

def deriv_bce_weight(y_pred,y_true,x):
     return np.dot((y_pred - y_true), x) / len(y_true)

def deriv_bce_bias(y_pred,y_true):
    return np.mean(y_pred - y_true)
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
    def __init__(self, input_layer, hidden_layer1, hidden_layer2, output_layer):
        self.w1 = np.random.normal(size=(input_layer, hidden_layer1))
        self.w2 = np.random.normal(size=(hidden_layer1, hidden_layer2))
        self.w3 = np.random.normal(size=(hidden_layer2, output_layer))

        self.b1 = np.random.normal(size=(hidden_layer1,))
        self.b2 = np.random.normal(size=(hidden_layer2,))
        self.b3 = np.random.normal(size=(output_layer,))

    def forward(self, x):
        self.l1,self.a1 = linear(x,self.w1,self.b1)
        self.l2,self.a2 = linear(self.l1,self.w2,self.b2)
        self.l3,self.a3 = linear(self.l2,self.w3,self.b3)
        self.output=sigmoid(self.l3)
        return self.output
    def backpropagation(self, x, y_true):
        m = y_true.shape[0]
        error = self.output - y_true  # Ensure `error` shape is (m, output_dim)

        d_w3 = np.dot(self.a2.T, error) / m  # Gradient for weights between second hidden and output
        d_b3 = np.sum(error, axis=0) / m  # Gradient for biases in the output layer

        # Backpropagate the error to the second hidden layer
        delta2 = np.dot(error, self.w3.T) * relu_derivative(self.l2)  # Should be (m, hidden_dim2)
        d_w2 = np.dot(self.a1.T, delta2) / m  # Gradient for weights between first and second hidden layers
        d_b2 = np.sum(delta2, axis=0) / m  # Gradient for biases in the second hidden layer

        # Backpropagate to the first hidden layer
        delta1 = np.dot(delta2, self.w2.T) * relu_derivative(self.l1)  # Should be (m, hidden_dim1)
        d_w1 = np.dot(x.T, delta1) / m  # Gradient for weights between input and first hidden layer
        d_b1 = np.sum(delta1, axis=0) / m  # Gradient for biases in the first hidden layer

        # Update weights and biases using gradient descent
        self.w3 -= learning_rate * d_w3
        self.b3 -= learning_rate * d_b3
        self.w2 -= learning_rate * d_w2
        self.b2 -= learning_rate * d_b2
        self.w1 -= learning_rate * d_w1
        self.b1 -= learning_rate * d_b1



    def fit(self, x_train, y_train):
        for epoch in range(epochs):
            y_pred = self.forward(x_train)
            loss = bce(y_train, y_pred)
            self.backpropagation(x_train, y_train)
            if epoch % 100 == 0:
                print(f"Epoch: {epoch} | Loss: {loss:.5f}")

        


model = BinaryClassificationModel(input_layer=2, hidden_layer1=4, hidden_layer2=4, output_layer=1)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)


model.fit(x_train, y_train)


new_y_pred= np.round(sigmoid(model.forward(x=x)))

print(new_y_pred[:10])


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