import numpy as np
import mnist
import matplotlib.pyplot as plt
from SimpleNetwork import SimpleNetwork
from ActivationFunctions import sigmoid, d_sigmoid
from LossFunctions import l2_loss, d_l2_loss
# Just for macOS plots
import matplotlib as mpl
mpl.use('macosx')


# ==============================================================================
# Main Call
# ==============================================================================

if __name__ == "__main__":
    np.random.seed(42)  # Fixing the seed for the random number generation, to get reproducable results.

    X_train, y_train = mnist.train_images(), mnist.train_labels()
    X_test, y_test = mnist.test_images(), mnist.test_labels()
    num_classes = 10  # classes are the digits from 0 to 9
    X_train, X_test = X_train.reshape(-1, 28 * 28), X_test.reshape(-1, 28 * 28)
    print("Pixel values between {} and {}".format(X_train.min(), X_train.max()))
    X_train, X_test = X_train / 255., X_test / 255.
    print("Normalized pixel values between {} and {}".format(X_train.min(), X_train.max()))
    # Finally, to compute the loss, we need to one-hot the labels,
    # e.g. converting the label 4 into [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    y_train = np.eye(num_classes)[y_train]
    # Let us use our SimpleNetwork class and instantiate a network with 2 hidden layers, taking a flattened image
    # as input and returning a 10-value vector representing its belief the image belongs to each of the class
    # (the highter the value,the stronger the belief):
    mnist_classifier = SimpleNetwork(num_inputs=X_train.shape[1],
                                     num_outputs=num_classes, hidden_layers_sizes=[64, 32],
                                     activation_func=sigmoid, d_activation_func=d_sigmoid,
                                     loss_func=l2_loss, d_loss_func=d_l2_loss)
    # Check how our network performs (computing its loss over the training set, and its accuracy over the test set)
    predictions = mnist_classifier.forward(X_train)  # forward pass
    loss_untrained = mnist_classifier.loss_function(predictions, y_train)  # loss computation

    accuracy_untrained = mnist_classifier.evaluate_accuracy(X_test, y_test)  # Accuracy
    print("Untrained : training loss = {:.6f} | val accuracy = {:.2f}%".format(
        loss_untrained, accuracy_untrained * 100))

    losses, accuracies = mnist_classifier.train(X_train, y_train, X_test, y_test,
                                                batch_size=30, num_epochs=500)
    # note: Reduce the batch size and/or number of epochs if your computer can't
    #       handle the computations / takes too long.
    #       Remember, numpy also uses the CPU, not GPUs as modern Deep Learning
    #       libraries do, hence the lack of computational performance here.

    losses, accuracies = [loss_untrained] + losses, [accuracy_untrained] + accuracies

    # ==============================================================================
    # PLOT SECTION
    # ==============================================================================
    fig, ax_loss = plt.subplots()

    color = 'red'
    ax_loss.set_xlim([0, 510])
    ax_loss.set_xlabel('Epochs')
    ax_loss.set_ylabel('Training Loss', color=color)
    ax_loss.plot(losses, color=color)
    ax_loss.tick_params(axis='y', labelcolor=color)

    ax_acc = ax_loss.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'blue'
    ax_acc.set_xlim([0, 510])
    ax_acc.set_ylim([0, 1])
    ax_acc.set_ylabel('Val Accuracy', color=color)
    ax_acc.plot(accuracies, color=color)
    ax_acc.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()
