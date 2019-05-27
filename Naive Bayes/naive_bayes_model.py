from numpy import *
from post_dataset import *

"""
Naive Bayes is a classification algorithm in machine learning. It aims at computing the probs of
every class given the input x, and then choose the class with largest probablity as the final result.
"""
class Naive_bayes_model(object):

    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels


    def train_model(self):
        size, columns = len(self.dataset), len(self.dataset[0])

        # Use dict to count the time every label occurs, then we can easily get all probs of 'p(Ci)'.
        labels_count = {}
        for label in self.labels:
            labels_count[label] = labels_count.get(label, 0) + 1
        for key in list(labels_count.keys()):
            labels_count[key] /= float(size)


        # Initialize the dict used to compute all the conditional probs of 'p(xi|Cj)'.
        time_dict = {}
        for key in list(labels_count.keys()):
            # 'time_dict.setdefault(key, {})['times'] = ones(columns)' can replace the following two lines:
            time_dict[key] = {} # avoid 'key error'
            time_dict[key]['times'] =  ones(columns) # Replace 0 to avoid prob*0
            time_dict[key]['dims'] = 1 # Replace 0 to avoid num/0


        # Compute every p(x|Ci)=p(x1|Ci)·p(x2|Ci)···p(xn|Ci)
        for i in range(size):
            post_vec = self.dataset[i]
            label = self.labels[i]
            dim_all = sum(post_vec)

            time_dict[label]['times'] += post_vec
            time_dict[label]['dims'] += dim_all

        for key in list(time_dict.keys()):
            prob = time_dict[key]['times'] / float(time_dict[key]['dims'])
            # Use 'log' to prevent underflow caused by the multi of too many small probs.
            time_dict[key]['probs'] = log(prob)
        self.labels_count = labels_count
        self.time_dict = time_dict



    def __call__(self, input_vec):
        best_prob = -inf; best_label = -1

        # Loop to compute every prob for any label, and choose the label with largest prob as final result.
        for label in list(self.labels_count.keys()):
            prob = sum(input_vec * self.time_dict[label]['probs']) + log(self.labels_count[label])
            print('prob: ', prob)
            if prob > best_prob:
                best_prob = prob
                best_label = label

        print('Predict: ', best_label, '\n')
        return best_label


if __name__ == '__main__':

    # Initialize dataset
    post_dataset = Post_dataset('data/posts.txt')
    post_dataset.init_dataset()

    # Initialize model
    model = Naive_bayes_model(post_dataset.dataset, post_dataset.labels)
    model.train_model()

    # Predict
    input1 = ['I', 'love', 'my', 'dog']
    input2 = ['stupid', 'dog', 'it', 'is', 'garbage']
    input_vec = post_dataset.data2vec(input2)
    result = model(input_vec)