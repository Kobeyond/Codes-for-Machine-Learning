
"""
Base class for dataset, including create_vocal and create_one-hot_dataset. These functions
are common to both email dataset and post dataset. Therefore, we can greatly improve the
reuse and simplicity of code by inheriting this basic class.
"""

class Base_dataset(object):

    def create_vocal(self, dataset_str):
        vocal = set([])
        for data in dataset_str:
            # Union between two assembles:
            vocal = vocal | set(data)
        self.vocal = list(vocal)


    # Convert a sample into an one-hot vector.
    def data2vec(self, data):
        vocal_len = len(self.vocal)
        # The multi below means the repeat of python list [0]: [0, 0, ..., 0]
        data_vec = [0] * vocal_len

        for word in data:
            if word in self.vocal:
                # Set-of-words model
                data_vec[self.vocal.index(word)] = 1

                # Bag-of-words model
                # data_vec[self.vocal.index(word)] += 1
            else:
                print('The word \'%s\' is not in the vocabulary!' % word)
        # Finally, we get the one-hot vector of this sample.
        return data_vec


    # Convert the whole dataset into a matrix, consisted of many one-hot vectors.
    def dataset2vec(self, dataset_str):
        dataset_vec = []
        for data in dataset_str:
            dataset_vec.append(self.data2vec(data))

        self.dataset = dataset_vec
