import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        #initializing the empty weight matrix
        self.W = np.zeros(out_features, in_features) #the first dimension is the row, second is the column. First dimension should be the the number of output neurons
        #initializing the empty bias matrix
        self.b = np.zeros(out_features, 1) #it should be the number of training data in the batch not the input size of one batch

    def forward(self, A):
        """
        Take the batch of input training data and take one step forward. 
        Takes the input example and multiplicates with the weights and adds the biases, generating output Z
        """
        self.A = np.array(A) 
        self.N = np.shape(A)[0] #get the number of rows, aka the number of elements in a batch (bahch size)
        Z = self.A @ np.transpose(self.W) + np.ones(self.N, 1) @ np.transpose(self.b)
        return Z

    def backward(self, dLdZ):
        dLdA = dLdZ @ self.W
        dLdW = np.transpose(dLdZ) @ self.A
        dLdb = np.transpose(dLdZ.T) @ np.ones(self.N, 1) 
        self.dLdW = dLdW
        self.dLdb = dLdb

        return dLdA
