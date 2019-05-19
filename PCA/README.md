# PCA
PCA(Principal Component Analysis) is a very powerful tool in machine learning, which is frequently used to compress and visualize data. On the one hand, compressing the data before training our machine learning model can greatly decrease the memory storage it needs and speed up the training process. On the other hand, by visualizing the compressed data with a 2-D or 3-D figure, we can easily find some important clues hidden in the data, and then create a effective model for it. 

## Data Compressing

Data compressing, which is also called dimensionality reduction, try to find the axis which remained the largest variance at first. And then find the second axis(vertical to other axises), with second largest variance. Finally, repeat the process above until you get N axises.

## Data Reconstructing
