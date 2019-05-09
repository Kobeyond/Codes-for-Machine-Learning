# K-means

K-means is a popular `cluster algorithm` in `unsupervised learning`, which means we don't know the label of any sample. It aims at deviding the whole dataset into k clusters, and making sure the total error is lowest.

<img width='950' height='353' src="https://github.com/Kobeyond/Codes-for-Machine-Learning/blob/master/K-means/data/kmeans.png"/>


Assume that we use `SSE`(sum of square error) to measure the performance of a cluster algorithm, it represents the total distances between every example to its cluster center. A smaller SSE represents a better dataset split:

<img width='250' height='80' src="https://github.com/Kobeyond/Codes-for-Machine-Learning/blob/master/K-means/data/formular.png"/>



- Standard K-means

First, we randomly initialize the positions of the k cluster centers, and then update them(cluster assignment & cluster movement) constantly until being stable. 

- Bisecting K-means

To `avoid local minimum` usually happened in K-means, bisecting K-means algorithm regards the whole dataset as a single cluster at first, and then constantly splits the cluster which causes smaller SSE into two sub-clusters, until there are k clusters in total.




