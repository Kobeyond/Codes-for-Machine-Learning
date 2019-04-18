import operator
from dataset import *

"""
A decision tree is a decision support tool that uses a tree-like model of decisions and their possible consequences.
When predicting an answer, you will get deeper and deeper until arriving at a leaf node(result).
"""
class Decision_tree_model(object):

    # use 'model(*args)' to predict
    def __call__(self, input, annotations, tree):
        root_str = list(tree.keys())[0]
        sub_tree = tree[root_str]
        anno_index = annotations.index(root_str)

        # check every branch
        for key in list(sub_tree.keys()):
            if input[anno_index] == key:
                # arrive at branch node
                if type(sub_tree[key]).__name__ == 'dict':
                    answer = self.__call__(input, annotations, sub_tree[key])
                # arrive at leaf node
                else:
                    answer = sub_tree[key]
        return answer


    @staticmethod
    def majorirty_cnt(labels):
        # get the major class in a list
        label_count = {}
        for label in labels:
            label_count[label] = label_count.get(label, 0) + 1
        sorted_count = sorted(label_count.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_count[0][0]


    # As we need to use sub set to build tree frequently, making it static is properer.
    # So that we no longer worry about the data member not to change.
    @staticmethod
    def create_tree(dataset, labels, annotations):
        # Condition1: all the datas belong to a same class, needn't to split it again, just return the class label.
        if labels.count(labels[0]) == len(labels):
            return labels[0]
        # Condition2 to stop: no feature remained to split, then return the major class label.
        if len(dataset) == 0:
            return Decision_tree_model.majorirty_cnt(labels)

        columns = len(dataset[0])
        my_dataset = Lenses_dataset(dataset, labels, annotations)
        best_index = my_dataset.get_best_feature_to_split()
        best_feature = annotations[best_index]
        decision_tree = {best_feature:{}}
        # remove the used feature
        annotations = [annotations[i] for i in range(columns) if i != best_index]

        feature_values = [example[best_index] for example in dataset]
        unique_values = set(feature_values)
        # Create the whole tree recursively.
        for value in unique_values:
            sub_set = my_dataset.split_dataset(best_index, value)
            decision_tree[best_feature][value] = Decision_tree_model.create_tree(sub_set.dataset, sub_set.labels, sub_set.annotations)
        return decision_tree


    @staticmethod
    def get_num_leaf(tree):
        num_leaf = 0
        root_str = list(tree.keys())[0]
        sub_dict = tree[root_str]
        for key in sub_dict.keys():
            # node of branch
            if type(sub_dict[key]).__name__ == 'dict':
                num_leaf += Decision_tree_model.get_num_leaf(sub_dict[key])
            # node of leaf
            else:
                num_leaf += 1
        return num_leaf

    @staticmethod
    def get_tree_depth(tree):
        root_str = list(tree.keys())[0]
        sub_dict = tree[root_str]
        sub_depth = []
        for key in sub_dict.keys():
            if type(sub_dict[key]).__name__ == 'dict':
                sub_depth.append(Decision_tree_model.get_tree_depth(sub_dict[key]))
            else:
                sub_depth.append(1)
        return 1 + max(sub_depth)



if __name__ == '__main__':
    my_set = load_lenses_data_from_file('data/lenses.txt')
    tree = Decision_tree_model.create_tree(my_set.dataset, my_set.labels, my_set.annotations)

    print(Decision_tree_model.get_num_leaf(tree))
    print(Decision_tree_model.get_tree_depth(tree))









