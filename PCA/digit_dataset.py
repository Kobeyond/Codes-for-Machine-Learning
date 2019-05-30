from numpy import *

"""
Custom class for digit dataset. Transform the txt file to a numpy matrix. And then, we will 
use SVD to decompose it and reconstruct it for a test.
"""
class Digit_dataset(object):

    # Create a 32*32 matrix for the digit.
    def __init__(self, file_name):
        digit = []
        fr = open(file_name)
        for line in fr.readlines():
            row = [int(line[i]) for i in range(32)]
            digit.append(row)
        self.digit = mat(digit)


    @staticmethod
    def change_matrix(digit_old, shreshold=0.8):
        digit_new = mat(zeros((32,32)))
        for i in range(32):
            for j in range(32):
                if digit_old[i, j] >= shreshold:
                    digit_new[i, j] = 1
                else:
                    digit_new[i, j] = 0
        return  digit_new


    # Note: Here we need to decompose the dataset directly, instead of decomposing the cov-matrix used in PCA.
    # So, call the svd() with dataset itself.
    def compress(self, num_singular):
        U, S, VT = linalg.svd(self.digit)
        Sigma = mat(zeros((num_singular, num_singular)))
        for i in range(num_singular):
            Sigma[i, i] = S[i]
        digit_new = U[:, :num_singular] * Sigma * VT[:num_singular, :]
        return digit_new


if __name__ == '__main__':
    my_digit = Digit_dataset('data/digit_3.txt')

    for k in range(8):
        digit_recon = my_digit.compress(k + 1)
        digit_recon_new = Digit_dataset.change_matrix(digit_recon)

        # Save the reconstructed figure in a txt file.
        digit_str = ''
        digit_list = digit_recon_new.tolist()
        for line in digit_list:
            for num in line:
                digit_str += str(int(num))
            digit_str += '\n'

        fw = open('data/%d_singulars.txt' % (k + 1), 'w')
        fw.write(digit_str)
        fw.close()

