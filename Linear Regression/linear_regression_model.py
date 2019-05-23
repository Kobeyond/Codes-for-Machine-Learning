from numpy import *
import matplotlib.pyplot as plt
from simple_dataset import *

"""
Linear regression algorithm is mainly used in regression problems, which means the value of result is continuous.
It aims at finding a best straight line(or surface) crossing the dataset, together with a minimal MSE. 
"""
class Linear_regression_model(object):

    def __init__(self, dataset, labels, alpha=0.001):
        self.dataset = dataset
        self.labels = labels
        self.alpha = alpha

    # make a prediction
    def __call__(self, input):
        return mat(input) * self.weights


    def gradient_descent(self, iters=100):
        dataset = mat(self.dataset)
        # convert list to a m*1 vector
        labels = mat(self.labels).T
        size, columns = shape(dataset)
        weights = zeros((columns, 1))
        total_error = []

        # update weights
        for i in range(iters):
            results = dataset * weights
            error = sum(multiply(results - labels, results - labels))
            total_error.append(error)
            # Note: the equation is the same as logistic regression, but they are different in theory.
            gradients = dataset.T * (results - labels)
            weights -= self.alpha * gradients
        self.weights = weights

        # draw learning curve
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(1, 1 + iters), total_error)
        plt.xlabel('iteration'); plt.ylabel('Total loss')
        plt.show()


    # Use the formular to get best weights directively, but it can be slow when facing a large dataset.
    def normal_equation(self):
        x = mat(self.dataset)
        y = mat(self.labels).T

        # Only if xTx is invertible, can we get xTx.Inverse()
        xTx = x.T * x
        assert linalg.det(xTx) != 0.0
        self.weights = xTx.I * x.T * y


    # Use additional eye matrix to make sure invertible, and use 'shrinkage' to filter useless data.
    def ridge_regression(self, lam=0.2):
        # after normalization, dataset is already mat.
        dataset = self.dataset
        labels = mat(self.labels).T
        size, columns = shape(dataset)

        # It can be regarded as an upgraded normal equation.
        matrix_inv = dataset.T * dataset + lam * eye(columns)
        assert linalg.det(matrix_inv) != 0.0
        self.weights = matrix_inv.I * (dataset.T * labels)
        return self.weights


    def draw_regression_line(self):
        # Plot all the training examples
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # Note: in order to support dataset[:, 1], we must change dataset from list to array.
        dataset = array(self.dataset)
        ax.scatter(dataset[:, 1], self.labels, s=5)

        # Draw regression line.
        x1 = arange(0, 1.1, 0.1)
        x_vec = [[1, x] for x in x1]
        y = mat(x_vec) * self.weights
        ax.plot(x1, y, c='red')

        plt.xlabel('X'); plt.ylabel('Y')
        plt.show()



if __name__ == '__main__':
    dataset = [[1, 2.2, 4], [1, 3.1, 4.3], [1, 4.1, 4.6], [1, 3.3, 4], [1, 2.5, 3.8]]
    labels = [1.5, 2.5, 3.5, 2.5, 1.7]

    model = Linear_regression_model(dataset, labels, 0.001)
    model.gradient_descent(iters=20)
    # model.normal_equation()
    print(model([1, 5.1, 5.0]))
