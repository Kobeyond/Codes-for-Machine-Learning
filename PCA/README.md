# PCA
`Principal Component Analysis` is a powerful tool in machine learning, which is frequently used to compress and visualize data. On the one hand, compressing the data before training our machine learning model can decrease the memory storage it needs and greatly speed up the training process. On the other hand, by visualizing the compressed data with a 2D or 3D figure, we can easily find some important clues hidden in the data, and then create a effective model for it. 

<img width='500' height='375' src="https://github.com/Kobeyond/Codes-for-Machine-Learning/blob/master/PCA/data/PCA_model.png"/>

## Data Compressing

Data compressing, which is also called dimensionality reduction, tries to find the first axis which remains the largest variance, and then finds the second axis(vertical to other axises), with second largest variance. 

Repeat the process above until you get K axises. Finally, all the axises(vectors) make up the d*K transform matrix. As a result, we can use `matrix multiplication X'= X·V` to convert data from n-dimensional to K-dimensional. 

Tip: Eigen vectors which correspond larger eigen values, are 'main eigen vectors'. Choosing vectors with large eigen values will keep as many variances as possible.

## Data Reconstructing

Suppose that we have reduce the dimensions of dataset X to K-dimensional, by using `X'= X·V`, while V is a d*K matrix consists of K d-dimensional vectors. We can reconstruct an approximate dataset by simply applying `X = X'·V^T`. 

Tip: If K equals N(orginal number of features), then the reconstructed dataset will be exactly the same as original dataset. The certification process is as follows:

Suppose matrix C is the covariance matrix of dataset X, then C is a real-symmetric matrix.

=> Matrix C must contain N linearly-independent eigen vectors.

=> Matrix V, which is composed of N eigen vetors(normalized already) above, must be an orthogonal matrix.

=> `V·V^T = V^T·V = E`. 

=> According to the combination law of matrix multiplication, `(X·V)·V^T = X·(V·V^T) = X·E = X`. So we can reconstruct original dataset  X by applying another matrix multiplication. 


## How to choose K

It's very important to choose a appropriate `K` when reducing dimensions: If K is too large, the improvement of data compressing is slim. However, if K is too small, too much critical information in data will be dropped, and little will be remained, which will greatly decrease the performance of our model.

<img width='500' height='365' src="https://github.com/Kobeyond/Codes-for-Machine-Learning/blob/master/PCA/data/choose_k_new.png"/>

So, it's quite troublesome to determine and choose a proper number of principal components. Usually, we choose the smallest k which keeps 90%(for example) of variance at least(α represents the information dropped):

<img width='450' height='104' src="https://github.com/Kobeyond/Codes-for-Machine-Learning/blob/master/PCA/data/formulars_variance.png"/>

Fortunately, the variance can be measured by the eigen values(or singular values) as follow:

<img width='280' height='116' src="https://github.com/Kobeyond/Codes-for-Machine-Learning/blob/master/PCA/data/singular.png"/>

Tip:

- It's ok to decompose the dataset using either `ED` or `SVD`, because the ED of square matrix is a special case of SVD. In addition, the eigen values and singular values of square matrix are exactly the same when the matrix is square.

- SVD is more computationally effective than ED when the dataset is very large. So in demo2, I decompose the dataset2 using SVD, and finally get the correct answer.
