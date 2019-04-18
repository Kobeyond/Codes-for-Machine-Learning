from dataset import *
from decision_tree_model import *
from tools.tree_plotter import *
from tools.storage import *


def test_lenses():

    my_set = load_lenses_data_from_file('data/lenses.txt')
    model = Decision_tree_model()

    # Comment these codes to test loaded tree from .pkl file:
    # tree = Decision_tree_model.create_tree(my_set.dataset, my_set.labels, my_set.annotations)
    # store_tree(tree, 'data/tree.pkl')

    # load and draw the tree
    tree = load_tree('data/tree.pkl')
    createPlot(tree)

    # predict:
    input = ['young', 'hyper', 'yes', 'normal']
    answer = model(input, my_set.annotations, tree)
    print('answer: ', answer)



if __name__ == '__main__':
    # test what you want
    test_lenses()