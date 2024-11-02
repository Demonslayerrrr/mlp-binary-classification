from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import numpy as np


n_samples = 1000

random_seed = np.random.seed(100)

x,y = make_circles(n_samples=n_samples,
                       noise=0.03,
                       random_state=random_seed)




# plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.RdYlBu)
# plt.show()


train_split = int(0.8*len(x))

x_train,y_train,x_test,y_test = x[:train_split], y[:train_split],x[train_split:],y[train_split:]


def relu(x):
    return np.maximum(0,x)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def mse(y,y_pred):
    return ((y - y_pred)**2).mean()

def deriv_mse(y,y_pred):
    return -2*(y-y_pred)
class BinaryClassificationModel:
    def __init__(self,layers_number):
        self.layers = {}

        for i in range(layers_number+1):
            self.layers[f'w{i}'] = np.random.randn()

        self.items = list(self.layers.items())


        self.x1, self.x2 = np.random.randn(),np.random.randn()

        self.biases = {}

        for i in range(layers_number+1):
            self.biases[f"b{i}"] = np.random.randn()
    def forward(self):
        pass
    def print_layers(self):
        return self.layers
    def print_biases(self):
        return self.biases
    def print_features(self):
        return self.x1,self.x2
model = BinaryClassificationModel(layers_number=2)

print(model.print_layers())
print(model.print_biases())
print(model.print_features())
