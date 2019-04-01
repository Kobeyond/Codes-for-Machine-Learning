import numpy as np


"""
A simple class of Neural network for binary classification, which use sigmod as activation function.
In addition, I use back-propagation algorithm to get all the derivatives, and feed them to gradient 
descent to optimize the neural network.
"""

class Neural_Network(object):

    def __init__(self, dim_in, dim_hid, dim_out, size):
        # Initialize the basic properties of network
        self.dim_in=dim_in
        self.dim_hid=dim_hid
        self.dim_out=dim_out
        self.size = size

        # Randomly initialize weights
        self.w1=np.random.randn(dim_in, dim_hid)
        self.w2=np.random.randn(dim_hid, dim_out)
        self.learning_rate=0.3

    def sigmod(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmod_derivative(self, z):
        # element-wise product
        return self.sigmod(z) * (1 - self.sigmod(z))


    """
    Note:
        When x and w1 are both matrixs, x.column must equals w1.rows, and then x.dot(w1)
        return the result of matrix multi: x*w1.
        While in contrast, x*y represent element-wise multi, which satisifies x.shape==y.shape
    """
    def forward_propagation(self, x, y):

        h=x.dot(self.w1)   # Affine, h: [size, dim_hid]
        h_sigmod=self.sigmod(h)   # Activate, h_sigmod: [size, dim_hid]

        z_y=h_sigmod.dot(self.w2)   # Affine, z_y: [size, dim_out]
        y_predict=self.sigmod(z_y)   # Activate, y^: [size, dim_out]

        # Sum the errors from all the output units of all the samples in the batch.
        loss=np.square(y_predict-y).sum()

        #Construct a dict to save all the intermediate results.
        values={'h':h, 'h_sigmod':h_sigmod, 'z_y':z_y, 'y_predict':y_predict, 'loss':loss}
        return values


    # Use the 'Chain Rule' to compute all partial derivatives of Loss to every variables.
    def back_propagation(self, x, y, values):

        grad_y = 2.0 * (values['y_predict'] - y)                         # grad_y^: [size, dim_out]
        grad_z_y = (self.sigmod_derivative(values['z_y'])) * grad_y      # grad_z_y: [size, dim_out]
        grad_w2 = (values['h_sigmod']).T.dot(grad_z_y)                   # grad_w2: [dim_hid, dim_out]

        grad_h_sigmod = grad_z_y.dot((self.w2).T)                        # grad_h_sigmod: [size, dim_hid]
        grad_h = (self.sigmod_derivative(values['h'])) * grad_h_sigmod   # grad_h: [size, dim_hid]
        grad_w1 = x.T.dot(grad_h)                                        # grad_w2: [dim_in, dim_hid]

        # Divide 'data set size' to get the average of partial derivatives of all examples.
        return grad_w1/float(self.size), grad_w2/float(self.size)



if __name__ == '__main__':

    dataset_size=64
    dim_in, dim_hid, dim_out= 100, 10, 1
    nn = Neural_Network(dim_in, dim_hid, dim_out, dataset_size)

    # Just to test the network, randomly initialize x(input) and y(labels)
    x = np.random.randn(dataset_size, dim_in)
    y = np.random.rand(dataset_size, dim_out)

    iterations=2000
    # Use Gradient Descent to update all parameters, until converge.
    # Finally, we get the optimal model with best weights.
    for i in range(iterations):
        values = nn.forward_propagation(x, y)
        grad_w1, grad_w2 = nn.back_propagation(x, y, values)
        nn.w1 -= nn.learning_rate * grad_w1
        nn.w2 -= nn.learning_rate * grad_w2
        print('iters:%d, loss:%f' % (i, values['loss']))




