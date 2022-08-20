import numpy as np
from FullyConnectedLayer import FullyConnectedLayer


class SimpleNetwork(object):
    """A simple fully-connected NN.

    Args:
        num_inputs (int): The input vector size / number of input values.
        num_outputs (int): The output vector size.
        hidden_layers_sizes (list): A list of sizes for each hidden layer to add to the network
        activation_func (callable): The activation function for all the layers
        d_activation_func (callable): The derivated activation function
        loss_func (callable): The loss function to train this network
        d_loss_func (callable): The derivative of the loss function, for back-propagation

    Attributes:
        layers (list): The list of layers forming this simple network.
        loss_function (callable): The loss function to train this network.
        derivated_loss_function (callable): The derivative of the loss function, for back-propagation.

    """

    def __init__(self, num_inputs, num_outputs,
                 activation_func, d_activation_func,
                 loss_func, d_loss_func,
                 hidden_layers_sizes=(64, 32)):
        super().__init__()
        # We build the list of layers composing the network, according to the provided arguments:
        layer_sizes = [num_inputs, *hidden_layers_sizes, num_outputs]
        self.layers = [
            FullyConnectedLayer(layer_sizes[i], layer_sizes[i + 1],
                                activation_func, d_activation_func)
            for i in range(len(layer_sizes) - 1)]

        self.loss_function = loss_func
        self.derivated_loss_function = d_loss_func

    def forward(self, x):
        """
        Forward the input vector through the layers, returning the output vector.

        Args:
            x (ndarray): The input vector, of shape `(batch_size, num_inputs)`.

        Returns:
            activation (ndarray): The output activation value, of shape `(batch_size, layer_size)`.
        """
        for layer in self.layers:  # from the input layer to the output one
            x = layer.forward(x)
        return x

    def predict(self, x):
        """
        Compute the output corresponding to input `x`, and return the index of the largest
        output value.

        Args:
            x (ndarray): The input vector, of shape `(1, num_inputs)`.

        Returns:
            best_class (int): The predicted class ID.
        """
        estimations = self.forward(x)
        best_class = np.argmax(estimations)
        return best_class

    def backward(self, dL_dy):
        """
        Back-propagate the loss hrough the layers (require `forward()` to be called before).

        Args:
            dL_dy (ndarray): The loss derivative w.r.t. the network's output (dL/dy).

        Returns:
            dL_dx (ndarray): The loss derivative w.r.t. the network's input (dL/dx).
        """
        for layer in reversed(self.layers):  # from the output layer to the input one
            dL_dy = layer.backward(dL_dy)
        return dL_dy

    def optimize(self, epsilon):
        """
        Optimize the network parameters according to the stored gradients (require `backward()`
        to be called before).

        Args:
            epsilon (float): The learning rate.
        """
        for layer in self.layers:             # the order doesn't matter here
            layer.optimize(epsilon)

    def evaluate_accuracy(self, x_val, y_val):
        """
        Given a dataset and its ground-truth labels, evaluate the current accuracy of the network.

        Args:
            x_val (ndarray): The input validation dataset.
            y_val (ndarray): The corresponding ground-truth validation dataset.

        Returns:
            accuracy (float): The accuracy of the network
                              (= number of correct predictions/dataset size).
        """
        num_corrects = 0
        for i in range(len(x_val)):
            pred_class = self.predict(x_val[i])
            if pred_class == y_val[i]:
                num_corrects += 1
        return num_corrects / len(x_val)

    def train(self, x_train, y_train, x_val=None, y_val=None,
              batch_size=32, num_epochs=5, learning_rate=1e-3, print_frequency=5):
        """
        Given a dataset and its ground-truth labels, evaluate the current accuracy of the network.

        Args:
            x_train (ndarray): The input training dataset.
            y_train (ndarray): The corresponding ground-truth training dataset.
            x_val (ndarray): The input validation dataset.
            y_val (ndarray): The corresponding ground-truth validation dataset.
            batch_size (int): The mini-batch size.
            num_epochs (int): The number of training epochs i.e. iterations over the whole dataset.
            learning_rate (float): The learning rate to scale the derivatives.
            print_frequency (int): Frequency to print metrics (in epochs).

        Returns:
            losses (list): The list of training losses for each epoch.
            accuracies (list): The list of validation accuracy values for each epoch.
        """
        num_batches_per_epoch = len(x_train) // batch_size
        do_validation = x_val is not None and y_val is not None
        losses, accuracies = [], []
        for i in range(num_epochs):  # for each training epoch
            epoch_loss = 0
            for b in range(num_batches_per_epoch):  # for each batch composing the dataset
                # Get batch:
                batch_index_begin = b * batch_size
                batch_index_end = batch_index_begin + batch_size
                x = x_train[batch_index_begin: batch_index_end]
                targets = y_train[batch_index_begin: batch_index_end]
                # Optimize on batch:
                predictions = self.forward(x)  # forward pass
                L = self.loss_function(predictions, targets)  # loss computation
                dL_dy = self.derivated_loss_function(predictions, targets)  # loss derivation
                self.backward(dL_dy)  # back-propagation pass
                self.optimize(learning_rate)  # optimization of the NN
                epoch_loss += L

            # Logging training loss and validation accuracy, to follow the training:
            epoch_loss /= num_batches_per_epoch
            losses.append(epoch_loss)
            if do_validation:
                accuracy = self.evaluate_accuracy(x_val, y_val)
                accuracies.append(accuracy)
            else:
                accuracy = np.NaN
            if i % print_frequency == 0 or i == (num_epochs - 1):
                print("Epoch {:4d}: training loss = {:.6f} | val accuracy = {:.2f}%".format(
                    i, epoch_loss, accuracy * 100))
        return losses, accuracies
