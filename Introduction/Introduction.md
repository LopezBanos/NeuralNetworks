## Neural Networks

### Two Input Neuron (Mathematical Approach)

Suppose we have a two input array, $\vec{x}$, with the corresponding weights, $\vec{w}$. and a bias term the add up, b.

$$
\begin{align}
\vec{x} = (x_{0}, x_{1})\\
\vec{w} = (w_{0}, w_{1})
\end{align}
$$

    The result of adding the inputs taking into account the weights and the bias term is,

$$
z = \vec{x}\cdot \vec{w} + b
$$

In order, to move on to the next neuron there is an activation function that check if a condition is satisfied. The most common activation functions are:

- **Sigmoid Function**:
  
  $$
  \sigma(z) = \frac{1}{1 + e^{-z}}
  $$

- **Hyperbolic tangent**:
  
  $$
  tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}
  $$

- **Rectified Linear Unit (ReLU)**:
  
  $$
  ReLU(z) = max(0,z)
  $$

One of the most important activation function is the ReLU. If we concatenate neurons with a linear behaviour, that will be equivalent to having one single neuron with linear parameters that could only describe linear processes. **Therefore we need a non-lineal activation function for describing complex processes.**

```python
import numpy as np


class Neuron:
    """
    A simple feed-forward artificial neuron.
    """

    def __init__(self, num_inputs, activation_func):
        """
        :param num_inputs: Number of inputs.
        :param activation_func: The activation function we want to choose.
        """
        # Randomly initializing the weight vector and bias value:
        self.W = np.random.rand(num_inputs)
        self.b = np.random.rand(1)
        self.activation_func = activation_func

    def forward(self, x):
        """
        Forward the input signal through the neuron.
        :param x: Array that contain the inputs
        :return: The scalar product of the inputs and weights and the addition of the bias term.
        """
        z = np.dot(x, self.W) + self.b
        return self.activation_func(z)

    # Method for debugging
    def show(self):
        print("The activation function selected: ", self.activation_func)
        print("The array containing the weights is : ", self.W)
```

### 

### Layers in NN

A neural network consist on multiple layers of neurons. A layer is called **dense** when each neuron of the new layer is connected to all the values (output from neuron) from the previous layer.

Suppose we have a first layer that consist on three neurons, {a, b, c}. Each neuron takes the same input (for simplicity 2D array). If we want to compact the procces in one fomulae:

$$
z_{layer1} = \vec{x} \cdot W + b
$$

In this case **W** is a matrix that contain all the weights coefficients,

$$

W =
\left(\begin{array}{cc} 
w_{a1} & w_{b1} & w_{c1}\\
w_{a2} & w_{b2} & w_{c2}
\end{array}\right)



$$

So, the activation function applied to this layer will have the form,

$$
\begin{align}
\vec{y} = f(\vec{x}) = (f(\vec{x_{a}}), f(\vec{x_{b}}), f(\vec{x_{c}}))
\end{align}
$$

The output of the activation function is used in the following layer as the input and so on.

Most of the times, we need a bunch of inputs to train our model, that is called **batch**.

### Training a Neural Network

Training a NN means to optimize the parameters for a task by using the available data. There are different strategies:

- **Supervised Learning**: The NN access to data (inputs) but also the ground truth labels. It use the labels for checking if the prediction (when training) has improved or not. It applies when the NN is doing a mapping between two modalities.

- **Unsupervised Learning**: The NN computs the loss only based on its inputs and outputs. This method is useful for clustering, and compression (compare the properties of the compress data with respect the original data).

- **Reinforcement Learning**: An agent has some actions (walk, jump, ...) and navigate through an environment. Once the list of actions has finished the agent reach an state. There are some states that bring rewards. These rewards serve as feedback for the NN. The NN would modify the actions in order to maximize the rewards.

#### Loss Functions

The goal of the loss function is to evaluate how well the network, with its current parameters, is performing. There are plenty of loss functions:

- **L2 loss** (mostly used in supervised learning):
  
  $$
  L_{2}(y, y^{true}) = \sum_{i} \left(y_{i}^{true} - y_{i} \right)^{2}
  $$

- **L1 loss** (computes the absolute difference between the vectors):
  
  $$
  L_{1}(y, y^{true}) = \sum_{i}|y_{i}^{true} - y_{i}|
  $$

- **Binary cross-entropy (BCE)** (Converts the predicted probabilities into a logarithmic scale before comparing them to the expected values):
  
  $$
  BCE(y, y^{true}) = \sum_{i} \left[-y_{i}^{true}log(y_{i}) + (1 - y_{i}^{true})log(1-y_{i})\right]
  $$

At each training iteration of the training process, the derivatives of the loss with respect to each parameter of the network are computed. These derivatives indicate which small changes to the parameters need to be applied (with a -1 coefficient since the gradient indicates the direction of increase of the function, while we want to minimize it).

To compute this derivatives with respect the weights of the k-layer, we need to apply the chain rule:

$$
\frac{dL}{dW_{k}} = \frac{dL}{dy_{k}}\frac{dy_{k}}{dW_{k}} = \frac{dL}{dy_{k}}\frac{dy_{k}}{dz_{k}}\frac{dz_{k}}{dW_{k}} = \frac{dL}{dx_{k + 1}}\frac{dy_{k}}{dz_{k}}\frac{d(\vec{W_{k}} \cdot \vec{x_{k}} + b_{k})}{dW_{k}}
$$

In a similar way we can compute the derivatives respect to bias term.

Once we know the loss derivatives with respect each parameter, it is just a matter of updating them accordingly:

$$
W_{k} = W_{k} - \epsilon \frac{dL}{dW_{k}}
$$

- $\epsilon$ : Learning rate that control how each parameter should be updated in each iteration. If the learning rate is large the NN learns faster, but the NN could miss the minimal loss due to big steps.

- **Epoch** : One iteration over the whole training set.
