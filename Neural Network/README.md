# Neural Network

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

