# PCA
PCA(Principal Component Analysis) is a powerful tool in machine learning, which is frequently used to compress and visualize data. On the one hand, compressing the data before training our machine learning model can decrease the memory storage it needs and greatly speed up the training process. On the other hand, by visualizing the compressed data with a 2-D or 3-D figure, we can easily find some important clues hidden in the data, and then create a effective model for it. 

## Data Compressing

Data compressing, which is also called dimensionality reduction, tries to find the first axis which remains the largest variance, and then finds the second axis(vertical to other axises), with second largest variance. 

Repeat the process above until you get K axises. Finally, all the axises(vectors) make up the d*K transform matrix. As a result, we can use `matrix multiplication` to convert data from n-dimensional to K-dimensional. 

Tip: Eigen vectors which correspond larger eigen values, are 'main eigen vectors'. Choosing vectors with large eigen values will keep as many variances as possible.

## Data Reconstructing

Suppose that we have reduce the dimensions of dataset X to K-dimensional, by using `X'= X·V`, while V is a d*K matrix consists of K d-dimensional vectors. We can reconstruct original dataset simply using `X = X'·V^T`. The certification process is as follows:

Suppose matrix C is the covariance matrix of dataset X, then C is a real-symmetric matrix.

=> Matrix C must contain n linearly-independent eigen vectors.

=> Matrix V, which is composed of K eigen vetors(normalized already) above, must be an orthogonal matrix.

=> `V·V^T = V^T·V = E`. 

=> According to the combination law of matrix multiplication, `(X·V)·V^T = X·(V·V^T) = X·E = X`. So we can reconstruct original data X by applying another matrix multiplication. 
