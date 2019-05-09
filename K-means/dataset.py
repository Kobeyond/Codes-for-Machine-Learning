import matplotlib.pyplot as plt
from numpy import *


class My_dataset(object):

    def __init__(self, file_name):
        fr = open(file_name)
        lines = fr.readlines()
        dataset = []
        for line in lines:
            row = line.strip().split()
            # Use map() to change string list to float list. But in Python3 map() return iterable,
            # so we need to convert it to list.
            data = list(map(float, row))
            dataset.append(data)
        self.dataset = dataset

    # Plot all the data in a figure
    def draw_scatter(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # Note: in order to support dataset[:, 1], we must change dataset from list to array.
        dataset = array(self.dataset)
        ax.scatter(dataset[:, 0], dataset[:, 1], s=30)
        plt.show()


if __name__ == '__main__':
    my_dataset = My_dataset('data/testSet.txt')
    my_dataset.draw_scatter()
