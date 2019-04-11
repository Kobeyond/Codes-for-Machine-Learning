from numpy import *
from os import listdir


"""
Custom class for handwriting recognization, from which we can get values and labels.

"""
class Handwriting_dataset(object):

    def __init__(self, dir_name):
        file_list = listdir(dir_name)
        self.size = len(file_list)

        dataset, labels = [], []
        for i in range(self.size):
            # file name: 'label_count.txt'. For example, '0_2.txt' means the second sample of digit 0.
            file_name = file_list[i]

            img_vec = Handwriting_dataset.img2vector(self, dir_name + file_name)
            dataset.append(img_vec)
            label = (file_name.split('.')[0]).split('_')[0]
            labels.append(label)

        self.dataset = array(dataset)
        self.labels = labels


    # Every img file consists of 32 rows and 32 columns, with element 0 or 1.
    # Now, convert it into a 1*1024 vector.
    def img2vector(self, file_name):
        img_vector = zeros((1024))
        fr = open(file_name)
        lines = fr.readlines()
        # copy the element row by row.
        for row in range(32):
            line = lines[row]
            for column in range(32):
                img_vector[32 * row + column] = int(line[column])
        return array(img_vector)


if __name__ == '__main__':

    training_set = Handwriting_dataset('handwriting_digits/trainingDigits/')
    test_set = Handwriting_dataset('handwriting_digits/testDigits/')

    print('training set size: %d' % training_set.size)
    print('test set size: %d' % test_set.size)
