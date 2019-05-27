from email_dataset import *
from naive_bayes_model import *
import random


def test_email_classification(test_size=10):

    # Initialize original dataset.
    email_dataset = Email_dataset('data/email')
    email_dataset.init_dataset()

    # Get k non-repeating random numbers to generate test set.
    test_index = random.sample(range(0, 50), test_size)
    # Reverse the list to avoid 'index out of range': delete list[i] from end to front.
    test_index.sort(reverse=True)
    test_emails = []; test_set = []; test_labels = []
    # Create test set.
    for i in test_index:
        test_emails.append(email_dataset.emails[i])
        email_dataset.emails.pop(i)

        test_set.append(email_dataset.dataset[i])
        email_dataset.dataset.pop(i)

        test_labels.append(email_dataset.labels[i])
        email_dataset.labels.pop(i)

    # Initialize and train model
    model = Naive_bayes_model(email_dataset.dataset, email_dataset.labels)
    model.train_model()

    # Test all the examples.
    error_count = 0.0
    for i in range(test_size):
        test_email = test_emails[i]
        test_vec = test_set[i]
        label = test_labels[i]
        result = model(test_vec)
        if label != result:
            error_count += 1
            print('Email: ', test_email)
            print('label: ', label)
            print('predict: ', result, '\n')
    print('The total error rate is %.2f%%' % ((error_count / test_size) * 100))



if __name__ == '__main__':

    test_email_classification(10)