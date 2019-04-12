from logistic_regression_model import *
from test_model_dataset import *
from horse_dataset import *


def test_simple_dataset():
    my_dataset = Test_model_dataset('data/testSet.txt')
    dataset = my_dataset.dataset
    labels = my_dataset.labels
    model = Logistic_regression_model(dataset, labels)

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
    train_set = my_train_dataset.dataset
    train_labels = my_train_dataset.labels

    my_test_dataset = Horse_dataset('data/horseColicTest.txt')
    test_set = my_test_dataset.dataset
    test_labels = my_test_dataset.labels

    # construct LR model, and optimize it.
    model = Logistic_regression_model(train_set, train_labels)
    # model.gradient_descent()
    model.SGD()

    # test every example in test set.
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