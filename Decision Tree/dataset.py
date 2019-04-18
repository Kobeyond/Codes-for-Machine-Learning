from math import log

"""
Custom dataset class for lenses. It provides some functions about getting shannon entropy, splitting dataset,
and so on, which makes it greatly convinent to bulid a decision tree algorithm.
"""
class Lenses_dataset(object):

    # Here, different from usual, we construct custom dataset using direct dataset and labels, instead of file name.
    # Because we need to construct sub dataset and sub labels frequently when build our decision tree.
    def __init__(self, dataset, labels, annotations):
        self.dataset = dataset
        self.labels = labels
        self.annotations = annotations

    # To support syntax like 'len(my_dataset)'.
    def __len__(self):
        return len(self.dataset)


    def get_shannon_ent(self):
        size = len(self.dataset)
        # Count the time every label occurs, to compute shannon entropy.
        label_count = {}
        for i in range(size):
            label = self.labels[i]
            label_count[label] = label_count.get(label, 0) + 1

        shannon = 0.0
        for key in label_count.keys():
            prob = label_count[key] / float(size)
            shannon -= prob * log(prob, 2)
        return shannon


    # Get sub set from dataset where dataset[axis] == value, remember feature[axis] is thrown.
    def split_dataset(self, axis, value):
        dataset = self.dataset
        size, columns = len(dataset), len(dataset[0])

        sub_dataset = []; sub_labels = [];
        for i in range(size):
            if dataset[i][axis] == value:
                data = dataset[i][:axis]
                data.extend(dataset[i][axis+1:])
                sub_dataset.append(data)
                sub_labels.append(self.labels[i])
        # Drop the label splited.
        sub_annotations = [self.annotations[i] for i in range(columns) if i != axis]
        sub_set = Lenses_dataset(sub_dataset, sub_labels, sub_annotations)
        return sub_set


    def get_best_feature_to_split(self):
        dataset = self.dataset
        labels = self.labels
        size, num_features = len(dataset), len(dataset[0])

        # Original entropy before split.
        base_entropy = self.get_shannon_ent()
        best_change = 0.0; best_feature_index = 0

        # Try to use every feature to split.
        for axis in range(num_features):
            feature_list = [sample[axis] for sample in dataset]
            unique_feature = set(feature_list)
            new_entropy = 0.0
            # Split dataset to sub sets, and compute the total entropy changed.
            for value in unique_feature:
                sub_set = self.split_dataset(axis, value)
                prob = len(sub_set) / float(size)
                new_entropy += prob * sub_set.get_shannon_ent()
            # If entropy changes most this time, then mark this feature as a candidate to split.
            change = base_entropy - new_entropy
            if change > best_change:
                # change > 0 means infor gain > 0, and the data is going to be more ordered.
                best_change = change
                best_feature_index = axis
        return best_feature_index


def load_lenses_data_from_file(file_name):
    # read data from file
    fr = open(file_name)
    dataset = []; labels = [];

    lines = fr.readlines()
    for line in lines:
        splits = line.split()
        dataset.append(splits[0:4])
        if len(splits[4:]) == 2:
            labels.append(splits[4] + ' ' + splits[5])  # 'no lenses'
        else:
            labels.append(splits[4])  # 'soft', 'hard'
    annotations = ['age', 'prescript', 'astigmatic', 'tearRate']
    return Lenses_dataset(dataset, labels, annotations)


if __name__ == '__main__':
    my_dataset = load_lenses_data_from_file('data/lenses.txt')
    print('best index:', my_dataset.get_best_feature_to_split())

