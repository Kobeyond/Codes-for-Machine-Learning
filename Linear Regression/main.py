from simple_dataset import *
from linear_regression_model import *


def test_model():

    simple_dataset = Simple_dataset('data/training_set.txt')
    training_set = simple_dataset.dataset
    training_labels = simple_dataset.labels

    # train my model
    model = Linear_regression_model(training_set, training_labels, alpha=0.0003)
    # model.gradient_descent(iters=50)
    model.normal_equation()
    model.draw_regression_line()

    # Evaluate the linear regression model on test set.
    test_dataset = Simple_dataset('data/test_set.txt')
    test_set = test_dataset.dataset
    test_labels = test_dataset.labels

    total_err = 0.0
    test_size, columns = shape(test_set)
    for i in range(test_size):
        predict = model(test_set[i])
        label = test_labels[i]
        total_err += (predict - label) ** 2
    print('The total MSE of %d test examples is: %f' % (test_size, total_err))


def test_lwlr_model(self, test_set, k=0.01):
        testset = mat(test_set)
        test_size = shape(testset)[0]

        answers = []
        total_err = 0.0
        for i in range(test_size):
            answer = self.__call__(test_set[i])
            label = self.labels[i]
            total_err += (answer - label) ** 2
        print()



if __name__ == '__main__':
    test_model()


