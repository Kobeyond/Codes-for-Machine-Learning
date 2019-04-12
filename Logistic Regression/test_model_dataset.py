import matplotlib
import matplotlib.pyplot as plt
from numpy import *


"""
Custom class for simple dataset, from which we can get dataset and labels in the form of lists.
"""
class Test_model_dataset(object):

    def __init__(self, file_name):
        fr = open(file_name)
        lines = fr.readlines()
        dataset = []; labels = []
        for line in lines:
            splits = line.split()
            dataset.append([1.0, float(splits[0]), float(splits[1])])
            labels.append(int(splits[2]))
        self.dataset = dataset
        self.labels = labels


    # Plot all the data in a figure
    def draw_scatter(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # Note: in order to support dataset[:, 1], we must change dataset from list to array.
        dataset = array(self.dataset)
        # Note: in order to realize element multi to labels, we must change it to array too, or it will repeat 15 times.
        # add 1 to every element: make sure spot size != 0 when label == 0
        labels = array(self.labels) + ones(shape(array(self.labels)))
        ax.scatter(dataset[:, 1], dataset[:, 2], 15.0 * labels, 15.0 * labels)
        plt.show()


if __name__ == '__main__':
    my_dataset = Test_model_dataset('data/testSet.txt')
    my_dataset.draw_scatter()

