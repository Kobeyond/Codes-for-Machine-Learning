import pickle


# store your decision tree(dict) in file.
def store_tree(tree, file_name):
    fw = open(file_name, 'wb')
    pickle.dump(tree, fw)
    fw.close()


# load tree(dict) from
def load_tree(file_name):
    fr = open(file_name, 'rb')
    return pickle.load(fr)