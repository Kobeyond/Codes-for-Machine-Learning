import matplotlib
import matplotlib.pyplot as plt
from numpy import *


"""
Custom class for horse colic dataset, from which we can get dataset and labels in the form of lists.
"""
class Horse_dataset(object):

    def __init__(self, file_name):
        fr = open(file_name)
        lines = fr.readlines()
        dataset = []; labels = []
        for line in lines:
            splits = line.split()
            data = [float(splits[i]) for i in range(len(splits) - 1)]
            # x0 = 1
            data.append(1.0)
            dataset.append(data)

            # get integer label(0/1) from str '0.000'/'1.000'
            num = splits[-1].split('.')[0]
            labels.append(int(num))
        self.dataset = dataset
        self.labels = labels



if __name__ == '__main__':

    my_dataset = Horse_dataset('data/horseColicTest.txt')
    print(my_dataset.dataset)
    print(my_dataset.labels)

