from numpy import *
import matplotlib.pyplot as plt


"""
Logistic Regression algorithm is mainly used to binary classification, which aims at finding a best decision boundary
to separate the training examples. While z >= 0 means positive (above the line), and z <= 0 means negative(beyond the
line). In addition, g(z) represents the probability of positive. 
"""
class Logistic_regression_model(object):

    # Note: dataset and labels should both be python lists.
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels
        self.learning_rate = 0.001

    def sigmod(self, x):
        # Note that x is a vector.
        return 1 / (1 + exp(-x))

    def __call__(self, input):
        # inner product between two vectors.
        z = mat(input) * self.weights
        prob = self.sigmod(z)
        if prob >= 0.5: return 1
        else: return 0


    def gradient_descent(self):
        dataset = mat(self.dataset)  # [size, columns]
        labels = mat(self.labels).transpose()  # [size, 1]
        size, columns = shape(self.dataset)
        weights = zeros((columns, 1))  # [columns, 1]

        # Use gradient descent to update the weights 500 times.
        for i in range(100000):
            z = dataset * weights
            probs = self.sigmod(z)  # [size, 1]

            # This is matrix mult between columns×size & size×1
            partial_derivatives = dataset.transpose() * (probs - labels.astype('float64'))   # [columns, 1]
            weights -= self.learning_rate * partial_derivatives  # [columns, 1]
        self.weights = weights


    def SGD(self):
        dataset = mat(self.dataset)
        size, columns = shape(dataset)
        weights = zeros((columns, 1))  # [columns, 1]

        # In SGD, every time we use the derivatives of i-th sample to represents the total.
        # Although, it may be somehow inaccurate, it greatly speed up iteration.
        for i in range(size):
            z = dataset[i] * weights
            prob = self.sigmod(z)

            # This is matrix mult between columns×1 & 1×1
            partial_derivatives = dataset[i].transpose() * (prob - self.labels[i])
            weights -= self.learning_rate * partial_derivatives
        self.weights = weights


    def draw_decision_boundary(self):
        size = shape(self.dataset)[0]
        x_cord0 = []; y_cord0 = []
        x_cord1 = []; y_cord1 = []

        for i in range(size):
            label = self.labels[i]
            # As x0=1, we only use x1 and x2 to plot spots.
            if label == 0:
                x_cord0.append(self.dataset[i][1])
                y_cord0.append(self.dataset[i][2])
            else:
                x_cord1.append(self.dataset[i][1])
                y_cord1.append(self.dataset[i][2])

        # Plot all the training examples.
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x_cord0, y_cord0, s=30, c='red', marker='x')
        ax.scatter(x_cord1, y_cord1, s=30, c='blue', marker='o')

        # Draw decesion boundary.
        x = arange(-3.0, 3.0, 0.1)
        y = -(self.weights[0] + self.weights[1] * x) / self.weights[2]
        ax.plot(x, y)

        plt.xlabel('X1'); plt.ylabel('X2')
        plt.show()



if __name__ == '__main__':
    # Initialize LR model.
    dataset = [[1, -0.017612, 14.053064], [1, -1.395634, 4.662541], [1, -0.752157, 6.538620], [1, -1.322371, 7.152853]]
    labels = [0, 1, 0, 0]
    model = Logistic_regression_model(dataset, labels)

    # Update weights.
    model.gradient_descent()
    # model.SGD()
    print('weights after gradient descent: {}'.format(model.weights))

    # Choose some labeled sample to test.
    result = model([1, 0.423363, 11.054677])
    print('result: %d' % result)

    model.draw_decision_boundary()




