from kNN_model import *
from dating_dataset import *
from handwriting_dataset import *


"""
Combine our kNN_model and dating_dateset to predict answer for dating.
rate: the proportion for deviding train & test set.
"""
def test_dating(rate):
    # a simple test for dating.
    my_dataset = Dating_dataset('dating.txt')
    my_dataset.normalize()

    # use dataset to construct my model.
    dataset = my_dataset.dataset
    labels = my_dataset.labels
    size = dataset.shape[0]
    test_count = int(size * rate)
    model = kNN_model(7, dataset[test_count:size, :], labels[test_count:size])

    # predict answer over every test sample.
    err_count = 0
    for i in range(test_count):
        input_vec = dataset[i, :]
        label = labels[i]
        answer = model(input_vec)
        if answer != label:
            err_count += 1
            print('  predict: %s, label: %s' % (answer, label))

    err_rate = err_count/float(test_count)
    print('The total rate of error is : %f' % err_rate)


# Predict answer over your input.
def predict_dating():
    # get inputs from keyboard
    print("How many miles you fly every year?")
    miles = float(input())
    print("How much ice cream you eat every week?")
    ice_cream = float(input())
    print("How long you play video games?")
    video_game = float(input())
    input_vec = array([miles, ice_cream, video_game])

    my_dataset = Dating_dataset('dating.txt')
    my_dataset.normalize()

    # use dataset to construct my model.
    dataset = my_dataset.dataset
    labels = my_dataset.labels
    model = kNN_model(7, dataset, labels)

    norm_input = (input_vec - my_dataset.min_value) / my_dataset.range
    answer = model(norm_input)
    print(answer)


# Combine our kNN_model and dateset to predict answer for handwriting recognization.
def test_handwriting():
    # Load traning set and test set
    my_training_set = Handwriting_dataset('handwriting_digits/trainingDigits/')
    training_set = my_training_set.dataset
    training_labels = my_training_set.labels

    my_test_set = Handwriting_dataset('handwriting_digits/testDigits/')
    test_set = my_test_set.dataset
    test_labels = my_test_set.labels

    # Use training set to construct kNN model
    model = kNN_model(5, training_set, training_labels)

    # Predict over every test sample.
    err_count = 0.0
    test_size = my_test_set.size
    for i in range(test_size):
        test_sample = test_set[i]
        answer = model(test_sample)
        label = test_labels[i]

        if answer != label:
            err_count += 1
            print('  predict: %s, label: %s' % (answer, label))

    err_rate = err_count / test_size
    print('The total error rate for handwriting is: %f' % err_rate)



 # You can test what you want here:
if __name__ == '__main__':
    # test_dating(rate=0.1)

    # predict_dating()

    test_handwriting()

