from numpy import *
import operator


"""
KNN algorithm aims at computing the distances between the input vector and all the training samples,
and predicts output as the most frequent label appeared in the range of the nearest k samples.
"""
class kNN_model(object):

    def __init__(self, k, dataset, labels):
        self.k = k
        self.dataset = dataset
        self.labels = labels


    def __call__(self, input_vec):
        assert len(input_vec) == len(self.dataset[0])
        size = (self.dataset).shape[0]

        # Element-wise operation to compute all distances simultanenously.
        diff_matrix = tile(input_vec, (size,1)) - self.dataset
        square_diff = diff_matrix ** 2
        distances = (square_diff.sum(axis=1)) ** 0.5
        sort_index = distances.argsort()

        # Use dict to count the time every label occurred.
        label_count = {}
        for i in range(self.k):
            label = (self.labels)[sort_index[i]]
            label_count[label] = label_count.get(label, 0) + 1
        sorted_count = sorted(label_count.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_count[0][0]


if __name__ == '__main__':

    # A simple test for our kNN model.
    dataset = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    model = kNN_model(3, dataset, labels)

    answer = model([0.2, 0.1])
    print('Predict answer is: %c' % answer)







