import matplotlib.pyplot as plt
from numpy import *

"""
Load simple dataset for linear regression.
Row in file: [feature1, feature2, ..., featureN, label]
And then, we can get matrix of dataset and labels conveniently.
"""
class Simple_dataset(object):

    def __init__(self, file_name):
        fr = open(file_name)
        dataset = []; labels = [];
        for line in fr.readlines():
            splits = line.split()
            num_features = len(splits)
            # The column in the end is the label of this line.
            data = [float(splits[i]) for i in range(num_features - 1)]
            label = float(splits[-1])

            dataset.append(data)
            labels.append(label)
        self.dataset = dataset
        self.labels = labels

    def normalize(self):
        dataset = mat(self.dataset)
        means = mean(dataset, 0)
        vars = var(dataset, 0)
        # operations between matrix and vector:
        self.dataset = (dataset - means) / vars


    # Plot all the data in a figure
    def draw_scatter(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.xlabel('X1'); plt.ylabel('Label')
        # Note: in order to support 'dataset[:, 1]', we must change dataset from list to array.
        dataset = array(self.dataset)
        ax.scatter(dataset[:, 1], self.labels, s=8)
        plt.show()


if __name__ == '__main__':
    my_dataset = Simple_dataset('data/training_set.txt')
    my_dataset.normalize()
    my_dataset.draw_scatter()

