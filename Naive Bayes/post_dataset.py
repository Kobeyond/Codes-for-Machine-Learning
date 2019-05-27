from numpy import *
from base_dataset import *

"""
Custom dataset for posts, we can convert all the posts to a matrix, every
post is replaced with a feature vector(one-hot vector) by the vocabulary.
"""
class Post_dataset(Base_dataset):

    # Read posts from a txt file.
    def __init__(self, filename):
        posts = []; labels = []
        fr = open(filename)
        for line in fr.readlines():
            row = line.strip().split()
            length = len(row)
            posts.append(row[ :length - 1])
            labels.append(int(row[-1]))
        self.posts = posts
        self.labels = labels


    # Call the function of the parent class, to generate self.vocal and self.dataset.
    def init_dataset(self):
        self.create_vocal(self.posts)
        self.dataset2vec(self.posts)


if __name__ == '__main__':
    post_dataset = Post_dataset('data/posts.txt')
    post_dataset.init_dataset()

    print('posts:\n', post_dataset.posts)
    print('labels:\n', post_dataset.labels)
    print('dataset: \n', post_dataset.dataset)
    print('vocabulary:\n', post_dataset.vocal)