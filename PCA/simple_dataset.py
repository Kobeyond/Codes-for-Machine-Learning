from numpy import *
import math


class Simple_dataset(object):

    def __init__(self, file_name, nan=False):
        fr = open(file_name)

        if nan == False:
            # Use list comprehension to create dataset
            dataset = [line.strip().split('\t') for line in fr.readlines()]
            dataset_float = [list(map(float, data)) for data in dataset]
        else:
            dataset = []
            for line in fr.readlines():
                splits = line.strip().split()
                length = len(splits)
                # Use a special value to replace 'NaN', then we can change the dataset to 'mat'.
                for i in range(length):
                    if splits[i] == 'NaN': splits[i] = '6.66666'

                dataset.append(splits)

            # Use mean values to replace the missing data at every column.
            dataset_float = mat([list(map(float, data)) for data in dataset])
            print(shape(dataset_float))
            columns = shape(dataset_float)[1]
            for j in range(columns):
                mean_val = mean(dataset_float[nonzero(dataset_float[:, j].A != 6.66666)[0], j])
                dataset_float[nonzero(dataset_float[: j].A == 6.66666)[0], j] = mean_val

        self.dataset = mat(dataset_float)



