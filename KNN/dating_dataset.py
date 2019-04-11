from numpy import *
import os
import matplotlib
import matplotlib.pyplot as plt

"""
Custom class for dating dataset, from which we can get values and labels.
sample: [flying miles, ice creams, video games]
label: largeDoses / smallDoses / didntLike
"""

class Dating_dataset(object):

    def __init__(self, file_name):

        fr = open(file_name)
        lines = fr.readlines()
        self.size = len(lines)

        # Construct dataset line by line.
        dataset = zeros((self.size, 3))
        labels = []
        for i in range(self.size):
            line = lines[i].strip().split('\t')
            dataset[i, :] = line[0:3]
            labels.append(line[-1])
        self.dataset = dataset
        self.labels = labels

    def normalize(self):
        """
        Normalize the dataset in to range [0, 1] by applying:
            value =(value - min) / (max - min)
        """
        min_value = self.dataset.min(0)
        max_value = self.dataset.max(0)
        range = max_value - min_value
        self.dataset = (self.dataset - tile(min_value, (self.size, 1))) / tile(range, (self.size, 1))
        self.min_value = min_value
        self.range = range


    def draw_scatter(self, axis1, axis2):

        # Construct num_labels to draw spots in different size and color.
        num_labels = []
        for label in self.labels:
            if label == 'largeDoses': num_labels.append(3)
            elif label == 'smallDoses': num_labels.append(2)
            else: num_labels.append(1)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter((self.dataset)[:, axis1], (self.dataset)[:, axis2], 15.0 * array(num_labels), 15.0 * array(num_labels))
        plt.show()


if __name__ == '__main__':

    # simple test for my dataset.
    dataset = Dating_dataset('dating.txt')
    dataset.normalize()
    dataset.draw_scatter(1, 2)

