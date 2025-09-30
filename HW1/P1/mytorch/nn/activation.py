import numpy as np
import scipy

class Activation:
    def forward(self, Z):
        """
        This function takes Z matrix which is the output of the previous layer and uses the activation function to return A. 
        Z matrix is of size N x Cout, N is the batch size and Cout is the number of features.
        """
        self.A = Z
        return self.A
    
    def backward(self, dLdA):
        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ         
        return dLdZ
    

class Sigmoid:
    def forward(self, Z):
        self.Z = Z
        self.A = 1/(1 + np.e ** - self.Z)
        return self.A

    def backward(self, dLdA):
        dLdZ = dLdA * (self.A - self.A * self.A) 
        return dLdZ


class Tanh:
    def forward(self, Z):
        self.Z = Z
        self.A = (np.e ** Z + np.e ** -Z)/(np.e ** Z - np.e ** -Z)
        return self.A

    def backward(self, dLdA):
        dLdZ = dLdA * (1-(np.e ** self.Z + np.e ** - self.Z)/(np.e ** self.Z - np.e ** - self.Z)**2)
        return dLdZ

class ReLU:
    def forward(self, Z):
        self.Z = Z
        self.A = np.maximum(0, self.Z)
        return self.A
    
    def backward(self, dLdA):
        return None
    
class GELU:
    def forward(self, Z):
        self.Z = Z
        self.A = 0.5 * Z * (1 + scipy.special.erf(Z / np.sqrt(2)))
        return self.A

    def backward():
        return None

class Softmax:
    def forward():
        return None
    
    def backward():
        None