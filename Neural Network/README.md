# Neural Network
Neural Network is a very powerful and widely used model in machine learning. It can be used to handle not only classification problems, but also regression problems.

Usually, neural network consists of one input layer, one ouotput layer and several hidden layers. Every layer is composed of some units which are called neurons. 

<img width='400' height='304' src="https://github.com/Kobeyond/Codes-for-Machine-Learning/blob/master/Neural%20Network/data/neural_network.png"/>


## Forward Propagation
We we train our neural network or make a prediction over a input, it's essential to pass the input from front to end. It is called `forward propagation`. 

<img width='920' height='188' src="https://github.com/Kobeyond/Codes-for-Machine-Learning/blob/master/Neural%20Network/data/forward_vectorized.png"/>



## Back Propagation
Similar to other machine learning algorithms, the core to train neural network is to get all the gradients of weights and biases, and then we can use gradient descent or other optimizers to train the model.

However, due to the complex structure of neural network, calculating gradients is somehow difficult. Then, the famous `back-propagation algorithm` occurs. We can easily get the gradients layer by layer, from back to front.

- Scalar BP

The `scalar` back propagation formulars are below:

<img width='250' height='274' src="https://github.com/Kobeyond/Codes-for-Machine-Learning/blob/master/Neural%20Network/data/scalar_bp.png"/>

In addition, the completed process of deduction is:

<img width='700' height='793' src="https://github.com/Kobeyond/Codes-for-Machine-Learning/blob/master/Neural%20Network/data/scalar_updated.png"/>

- Vectorized BP

Actually, when we train the network, it's more computationally efficient to get the gradient vectors of weight and bias directly, by using `vectorized BP formulars`, rather than use 'for' loop to get it one by one. The vectorized formulars are below:

<img width='300' height='216' src="https://github.com/Kobeyond/Codes-for-Machine-Learning/blob/master/Neural%20Network/data/vectorized_bp.png"/>


Similarly, the completed process of deduction is:

<img width='950' height='455' src="https://github.com/Kobeyond/Codes-for-Machine-Learning/blob/master/Neural%20Network/data/vectorized.png"/>

- Extension

Up to now, the vectorized formulars above are based on a single example, because we suppose that input x is a vector. To train our neural network, what we need is to use these formulars to all the examples. And then sum up all the gradients because every iteration of training is based on the whole dataset(Standard Gradient Descent). However, it can be slow to use a 'for' loop over every example.

Therefore, in order to fit the `2-D input`(dataset), we can vectorize the formulars again, which can greatly speed up training process. Then, the variables X, Z, a, y become 2-D matrixes. The formulars are these:

