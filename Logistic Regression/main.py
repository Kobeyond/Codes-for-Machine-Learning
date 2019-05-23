from logistic_regression_model import *
from test_model_dataset import *
from horse_dataset import *


def test_simple_dataset():
    my_dataset = Test_model_dataset('data/testSet.txt')
    model = Logistic_regression_model(my_dataset.dataset, my_dataset.labels)

    # Update weights.
    model.gradient_descent()
    print('weights after gradient descent: {}'.format(model.weights))

    # Choose some labeled sample to test.
    result = model([1, 0.423363, 11.054677])
    print('result: %d' % result)

    model.draw_decision_boundary()


def test_horse_colic():
    # create training set and test set.
    my_train_dataset = Horse_dataset('data/horseColicTraining.txt')
    my_test_dataset = Horse_dataset('data/horseColicTest.txt')
    test_set = my_test_dataset.dataset
    test_labels = my_test_dataset.labels

    # Create LR model, and optimize it.
    model = Logistic_regression_model(my_train_dataset.dataset, my_train_dataset.labels)
    model.gradient_descent(100000)
    # model.SGD()

    # Test every example in test set.
    err_count = 0.0
    test_size = len(test_set)
    for i in range(test_size):
        answer = model(test_set[i])
        label = test_labels[i]
        if answer != label:
            err_count += 1
            print('Error! Predict: %s, label:%s' % (answer, label))
    err_rate = err_count / test_size
    print('The total error rate on test set is: %f' % err_rate)


if __name__ == '__main__':

    # test_simple_dataset()

    test_horse_colic()
