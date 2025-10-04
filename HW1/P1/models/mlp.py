import numpy as np

from mytorch.nn.linear import Linear
from mytorch.nn.activation import ReLU
class MLP0:
    def __init__(self):
        self.layers = [Linear(2, 3), ReLU()]


    def forward(self, A0):
        Z0 = self.layers[0].forward(A0)
        A1 = self.layers[1].forward(Z0)
        return A1

    
    def backward(self, dLdA1):
       dLdZ0 = self.layers[1].backward(dLdA1)
       dLdA0 = self.layers[0].backward(dLdZ0)
       return dLdA0 
    
class MLP2:
    def __init__(self):
        self.layers = [Linear(2, 3), ReLU(), Linear(3, 2), ReLU()]

    def forward(self, A0):
        A1 = self.layers[1].forward(self.layers[0].forward(A0))
        A2 = self.layers[3].forward(self.layers[2].forward(A1))
        return A2

    def backward(self, dLdA2):
        dLdA1 = self.layers[2].backward(self.layers[3].backward(dLdA2))
        dLdA0 = self.layers[0].backward(self.layers[1].backward(dLdA1))
        return dLdA0
    

class MLP4:
    def __init__(self):
        self.layers = [Linear(2, 4), ReLU(), Linear(4, 8), ReLU(), Linear(8, 8), ReLU(), Linear(8, 4), ReLU(), Linear(4, 2), ReLU()]

    def forward(self, A):
        for i in range(len(self.layers)):
            A = self.layers[i].forward(A)
        return A
    
    def backward(self, dLdA):
        for i in reversed(range(len(self.layers))):
            dLdA = self.layers[i].backward(dLdA)

        return dLdA