import matplotlib.pyplot as plt
from numpy import *


"""
Load simple dataset for linear regression.
"""
class Simple_dataset(object):

    def __init__(self, file_name):
        fr = open(file_name)
        dataset = []; labels = [];
        lines = fr.readlines()
        for line in lines:
            splits = line.split()
            data = [float(splits[0]), float(splits[1])]
            label = float(splits[-1])

            dataset.append(data)
            labels.append(label)
        self.dataset = dataset
        self.labels = labels

    # Plot all the data in a figure
    def draw_scatter(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # Note: in order to support dataset[:, 1], we must change dataset from list to array.
        dataset = array(self.dataset)
        ax.scatter(dataset[:, 1], self.labels, s=8)
        plt.show()


if __name__ == '__main__':
    my_dataset = Simple_dataset('data/training_set.txt')
    my_dataset.draw_scatter()

