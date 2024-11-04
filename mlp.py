from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import numpy as np





n_samples = 1000

epochs = 1000

learning_rate = 0.01

random_seed = np.random.seed(100)

x,y = make_circles(n_samples=n_samples,
                       noise=0.03,
                       random_state=random_seed)



# plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.RdYlBu)
# plt.show()


train_split = int(0.8*len(x))

x_train,y_train,x_test,y_test = x[:train_split], y[:train_split],x[train_split:],y[train_split:]


def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def accuracy_fn(y_true,y_pred):
  correct = np.equal(y_true,y_pred).sum().item()
  acc = (correct)/len(y_pred)*100

  return acc

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

