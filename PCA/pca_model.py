from numpy import *
from simple_dataset import *
import matplotlib.pyplot as plt


class PCA_model(object):

    def __init__(self, dataset):
        self.dataset = mat(dataset)
        # Use mean normalization, otherwise the reconstructed data will be slided as a whole.
        self.mean_val = mean(dataset, axis=0)
        self.dataset_mean = self.dataset - self.mean_val
        self.cov_array = cov(self.dataset_mean, rowvar=0)


    # Get eigen values and vectors of the matrix.
    @staticmethod
    def get_eigen(matrix):
        eig_vals, eig_vects = linalg.eig(mat(matrix))
        return eig_vals, eig_vects

    # Get singular values and vectors of the matrix.
    @staticmethod
    def get_SVD(matrix):
        # Note: decompose the cov-matrix as usual, when we need to PCA.
        # If we just want to decompose the matrix, apply svd to dataset directly.
        U, S, VT = linalg.svd(mat(matrix))
        return U, S, VT


    # Determine best number of principal components according to eigen values.
    def get_best_K(self, matrix, threshold=0.8, method='ED'):
        assert method in ['ED', 'SVD']

        if method == 'ED':
            eig_vals, eig_vects = PCA_model.get_eigen(matrix)
            # Get the indexes of the top N largest eigen values.
            eig_index = argsort(eig_vals) # small to large

            value_sum = 0.0; value_total = sum(eig_vals)
            for i in range(len(eig_vals)):
                # sum the top-i biggest value
                value_sum += eig_vals[eig_index[-i-1]]
                rate = value_sum / value_total
                print('Variance remained(K=%d): %.3f' % (i+1, rate))
                if rate >= threshold: return i+1
        else:
            U, S, VT = PCA_model.get_SVD(matrix)
            value_sum = 0.0; value_total = sum(S)
            for i in range(len(S)):
                # The singular values in matrix S have been arranged from largest to smallest
                value_sum += S[i]
                rate = value_sum / value_total
                print('Variance remained(K=%d): %.3f' % (i + 1, rate))
                if rate >= threshold: return i+1


    # Convert the dataset from N-dim to K-dim with PCA.
    def dim_reduce(self, matrix, K, method='ED'):
        assert method in ['ED', 'SVD']

        # In both situations, we decompose cov-matrix of dataset, bacause we want to apply PCA.
        if method == 'ED':
            eig_vals, eig_vects = PCA_model.get_eigen(matrix)
            # Get the indexes of the top K largest eigen values.
            eig_index = argsort(eig_vals)
            eig_index = eig_index[: -(K + 1): -1]

            # Get the transform matrix (consists of K eigen vectors above), and get the dim-reduced dataset.
            transform_mat = eig_vects[:, eig_index]
            self.dataset_new = self.dataset_mean * transform_mat
            self.dataset_rebuild = self.dataset_new * transform_mat.T + self.mean_val
        else:
            # U, S, VT: [n, n], n is number of original features.
            U, S, VT = PCA_model.get_SVD(matrix)

            # Use U(n*k), S(k*k) and VT(k*n) to reduce and reconstruct dataset.
            transform_mat = U[:, :K]
            self.dataset_new = self.dataset_mean * transform_mat
            self.dataset_rebuild = self.dataset_new * transform_mat.T + self.mean_val

        print('shape_reduced:', shape(self.dataset_new))
        print('shape_rebulid:', shape(self.dataset_rebuild))


    def draw_dataset(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(self.dataset[:, 0].flatten().A[0], self.dataset[:, 1].flatten().A[0], marker='^', s=20)
        ax.scatter(self.dataset_rebuild[:, 0].flatten().A[0], self.dataset_rebuild[:, 1].flatten().A[0], marker='^', s=20, c='red')
        plt.xlabel('X1'); plt.ylabel('X2')
        plt.show()


    def draw_variance_K(self, iters):
        variance = []
        U, S, VT = PCA_model.get_SVD(self.cov_array)
        value_sum = 0.0; value_total = sum(S)
        for i in range(len(S)):
            # sum the top-i biggest value
            value_sum += S[i]
            variance.append(value_sum / value_total)

        fig = plt.figure()
        plt.plot(range(1, 1+iters), variance[:iters])
        plt.xlabel('K'); plt.ylabel('Variance Remained')
        plt.show()


if __name__ == '__main__':
    # simple_dataset = Simple_dataset('data/testSet.txt', nan=False)
    simple_dataset = Simple_dataset('data/secom.data', nan=True)
    model = PCA_model(simple_dataset.dataset)

    threshold = 0.95
    best_K = model.get_best_K(model.cov_array, threshold, method='ED')
    model.draw_variance_K(best_K)
    print('threshold: %.2f, best K: %d' % (threshold, best_K))
    model.dim_reduce(model.cov_array, best_K, method='ED')

    # model.draw_dataset()