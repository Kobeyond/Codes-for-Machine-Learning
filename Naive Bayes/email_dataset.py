from base_dataset import *
import re

"""
Custom class for email classification. We need to transform every email into an one-hot vector,
and then use naive bayes model to classify it. 
"""
class Email_dataset(Base_dataset):

    # Read all the emails from a dir.
    def __init__(self, dir_name):
        emails = []; labels = []
        for i in range(1, 26):
            # Ham
            email_ham = open(dir_name + '/ham/%d.txt' % i).read()
            emails.append(Email_dataset.text_parse(email_ham))
            labels.append(0)
            # Spam
            email_spam = open(dir_name + '/spam/%d.txt' % i).read()
            emails.append(Email_dataset.text_parse(email_spam))
            labels.append(1)

        self.emails = emails
        self.labels = labels


    @ staticmethod
    def text_parse(email_str):
        # Preprocess the email using regular expression, which will enhance the perfomance of our model.
        token_list = re.split(r'\W*', email_str)
        return [tok.lower() for tok in token_list if len(tok) > 2]


    # Call the function of the parent class, to generate self.vocal and self.dataset.
    def init_dataset(self):
        self.create_vocal(self.emails)
        self.dataset2vec(self.emails)


if __name__ == '__main__':
    email_dataset = Email_dataset('data/email')
    email_dataset.init_dataset()

    print('emails:\n', email_dataset.emails)
    print('labels:\n', email_dataset.labels)
    print('dataset: \n', email_dataset.dataset)
    print('vocabulary:\n', email_dataset.vocal)
