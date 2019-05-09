from numpy import *
from dataset import *
import matplotlib.pyplot as plt


class K_means_model(object):

    def __init__(self, dataset, k):
        self.dataset = array(dataset)
        self.k = k

    @staticmethod
    def get_square_distance(vec1, vec2):
        # sum up square distance from every axis.
        return sqrt(sum(power(vec1 - vec2, 2)))

    # Create a k*columns matrix, which represents the coordinates of k centers.
    def get_rand_centers(self):
        size, columns = self.dataset.shape
        min_matrix = tile(self.dataset.min(0), (self.k, 1))
        max_matrix = tile(self.dataset.max(0), (self.k, 1))
        range_matrix = max_matrix - min_matrix
        rand_matrix = random.rand(self.k, columns)
        # Make sure every coordinate of every center is in range [min, max]：
        center_points = min_matrix + multiply(range_matrix, rand_matrix)
        return mat(center_points)


    # Use k-means algorithm to get k clusters among examples.
    def start_cluster(self, dataset, K, get_distance, get_centers=get_rand_centers):
        dataset = mat(dataset)
        size, columns = shape(dataset)
        centers = get_centers(self)
        # column1: its cluster number, column2: error(distance ** 2)
        assignment = mat(zeros((size, 2)))
        cluster_change = True

        while cluster_change:
            cluster_change = False
            # Cluster assignment: assign every sample to the nearest center.
            for i in range(size):
                min_dist = inf; min_center = -1
                # search for the nearnest center
                for k in range(K):
                    distance = get_distance(dataset[i, :], centers[k, :])
                    if distance < min_dist:
                        min_center = k
                        min_dist = distance
                # As long as the center changes, it is unstable. In addition, a point whose cluster number doesn't change
                # but distance changes, is definitely casued by another point whose cluster center changed.
                if assignment[i, 0] != min_center:
                    cluster_change = True
                assignment[i, :] = min_center, min_dist ** 2

            # Cluster movement: move every center point to its center position.
            for k in range(K):
                # nonzero() return the positions of the data which equals k. The first
                # and second returned values represent the its position of row and column.
                sample_index = nonzero(assignment[:, 0].A == k)[0]
                sample_list = dataset[sample_index]
                centers[k, :] = mean(sample_list, axis=0)
        return assignment, centers


    # Use bi-Kmeans to avoid local maximum.
    def bi_kmeans(self, get_distance):
        dataset = mat(self.dataset)
        size, column = shape(dataset)
        cluster_assignment = mat(zeros((size, 2)))

        # At first, the total dataset belongs to a single cluster.
        center0 = mean(dataset, axis=0).tolist()[0]
        center_list = [center0]
        for i in range(size):
            cluster_assignment[i, 1] = get_distance(dataset[i, :], center0) ** 2

        # Next, split the big cluster until there are k clusters already.
        while len(center_list) < self.k:
            lowest_SSE = inf
            # Find the best cluster to devide.
            for i in range(len(center_list)):
                sub_dataset = dataset[nonzero(cluster_assignment[:, 0].A == i)[0], :]
                new_assignment, new_centers = self.start_cluster(sub_dataset, 2, get_distance)
                SSE_new = sum(new_assignment[:, 1])
                SSE_remained = sum(cluster_assignment[nonzero(cluster_assignment[:, 0].A != i)[0], 1])
                # Save the best information to split clusters:
                if SSE_new + SSE_remained < lowest_SSE:
                    best_center_index = i
                    best_new_centers = new_centers
                    best_new_assignment = new_assignment.copy()
                    lowest_SSE = SSE_new +SSE_remained

            # Now, split the original clusters according to the best info:
            # Notice the indexes of the 2 sub cluster.
            new_cluster_index = len(center_list)
            best_new_assignment[nonzero(best_new_assignment[:, 0].A == 1)[0], 0] = new_cluster_index
            best_new_assignment[nonzero(best_new_assignment[:, 0].A == 0)[0], 0] = best_center_index

            # Update the positions of the 2 new clusters.
            center_list[best_center_index] = best_new_centers[0, :].tolist()[0]
            center_list.append(best_new_centers[1, :].tolist()[0])

            # Update the assignment matrix by directly using sub-assignment matrix, which includes infors
            # of the 2 sub-clusters at the same time.
            cluster_assignment[nonzero(cluster_assignment[:, 0].A == best_center_index)[0], :] = best_new_assignment

        self.cluster_assignment = cluster_assignment
        self.center_list = mat(center_list)
        return cluster_assignment, mat(center_list)


    def draw_figure(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        markers = ['s', '8', 'v', 'p', '^', 'd', '>', 'h', '<']
        plt.xlabel('X1'); plt.ylabel('X2')

        # Plot samples in different clusters with different markers.
        for k in range(self.k):
            sub_dataset = mat(self.dataset)[nonzero(self.cluster_assignment[:, 0].A == k)[0], :]
            marker_style = markers[k % len(markers)]
            ax.scatter(sub_dataset[:, 0].flatten().A[0], sub_dataset[:, 1].flatten().A[0], marker=marker_style, s=30)

        # Plot all the centers.
        ax.scatter(self.center_list[:, 0].flatten().A[0], self.center_list[:, 1].flatten().A[0], marker='+', s=300, c='red')
        plt.show()



if __name__ == '__main__':
    my_dataset = My_dataset('data/testSet.txt')
    dataset = my_dataset.dataset
    model = K_means_model(dataset, 4)

    # assignment, centers = model.start_cluster(model.dataset, model.k, K_means_model.get_square_distance)
    bi_assignment, bi_centers = model.bi_kmeans(K_means_model.get_square_distance)
    print('assignment:\n', bi_assignment)
    print('centers:\n', bi_centers)

    model.draw_figure()


