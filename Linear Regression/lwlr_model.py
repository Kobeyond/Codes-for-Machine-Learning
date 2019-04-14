from numpy import *
from simple_dataset import *
import matplotlib.pyplot as plt


"""
LWLR algorithm is an upgraded algorithm. To avoid usual underfit, LWLR includes local weights W to measure the distance
between input and every example. So when it comes to predict answer, matrix W takes part in computation.
Shortcoming: Every time when we predict, the total dataset is necessary.
"""
class LW_linear_regression_model(object):

    def __init__(self, dataset, labels, k=0.01):
        self.dataset = dataset
        self.labels = labels
        self.k = k

    # Every time when we predict, the total dataset is necessary.
    def __call__(self, input):
        dataset = mat(self.dataset)
        labels = mat(self.labels).T
        size = shape(dataset)[0]
        # Use a diagonal matrix to save local weights for computational convience.
        local_weights = mat(eye(size))

        # Compute the distance(local weight) between input and every example.
        for i in range(size):
            # distance: x0^2 + x1^2 + x2^2
            diff_matrix = mat(input - dataset[i])
            distance = diff_matrix * diff_matrix.T
            # Note: do not use [i][i]
            local_weights[i, i] = exp(distance / (-2.0 * self.k * self.k))

        # Use the formular to compute optimal weights.
        matrix_inverse = dataset.T * local_weights * dataset
        assert linalg.det(matrix_inverse) != 0.0
        self.weights = matrix_inverse.I * dataset.T * (local_weights * labels)
        # Convert 1*1 matrix to a number.
        result = float((input * self.weights)[0][0])
        return result


    def draw_regression_line(self):
        # Plot the training examples.
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # Note: in order to support dataset[:, 1], we must change dataset from list to array.
        dataset = array(self.dataset)
        ax.scatter(dataset[:, 1], self.labels, s=4)

        # Draw regression line
        predicts = []
        steps = arange(0, 1, 0.01)
        for step in steps:
            x = [1.0, step]
            predict = self.__call__(x)
            predicts.append(predict)
        ax.plot(steps, predicts, c='red')
        plt.xlabel('X1'); plt.ylabel('Y')
        plt.show()


if __name__ == '__main__':
    my_dataset = Simple_dataset('data/training_set.txt')
    dataset = my_dataset.dataset
    labels = my_dataset.labels

    model = LW_linear_regression_model(dataset, labels, k=0.03)
    model.draw_regression_line()
    print(model.weights)


