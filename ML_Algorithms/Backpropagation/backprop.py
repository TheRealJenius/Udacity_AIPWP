import numpy as np
from data_prep import features, targets, features_test, targets_test

np.random.seed(21)

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))

def sig_prime(x): # defining sigmoid prime
    return sigmoid(x) * (1 - sigmoid(x))


# Hyperparameters
n_hidden = 2  # number of hidden units
epochs = 900
learnrate = 0.005

n_records, n_features = features.shape
last_loss = None
# Initialize weights
weights_input_hidden = np.random.normal(scale=1 / n_features ** .5,
                                        size=(n_features, n_hidden)) # the same as 1/math.sqrt(n)
weights_hidden_output = np.random.normal(scale=1 / n_features ** .5,
                                         size=n_hidden)

for e in range(epochs):
    del_w_input_hidden = np.zeros(weights_input_hidden.shape)
    del_w_hidden_output = np.zeros(weights_hidden_output.shape)
    for x, y in zip(features.values, targets):
        ## Forward pass ##
        # TODO: Calculate the output
        hidden_input = np.dot(x, weights_input_hidden) # this way around so it is (360,6) * (6,2)
        hidden_output = sigmoid(hidden_input)
        output = sigmoid(weights_hidden_output * hidden_output)

        ## Backward pass ##
        # TODO: Calculate the network's prediction error
        error = (y - output)

        # TODO: Calculate error term for the output unit
        output_error_term = error * sig_prime(weights_hidden_output * hidden_output)

        ## propagate errors to hidden layer

        # TODO: Calculate the hidden layer's contribution to the error
        hidden_error = error / weights_hidden_output
        
        # TODO: Calculate the error term for the hidden layer
        hidden_error_term = weights_input_hidden * output_error_term * sig_prime(hidden_input)
        
        # TODO: Update the change in weights
        del_w_hidden_output += learnrate * output_error_term * hidden_output
        del_w_input_hidden += learnrate * hidden_error_term * x[:,None]

    # TODO: Update weights  (don't forget to division by n_records or number of samples)
    weights_input_hidden += del_w_input_hidden / n_features
    weights_hidden_output += del_w_hidden_output / n_features

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        hidden_output = sigmoid(np.dot(x, weights_input_hidden))
        out = sigmoid(np.dot(hidden_output,
                             weights_hidden_output))
        loss = np.mean((out - targets) ** 2)

        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
hidden = sigmoid(np.dot(features_test, weights_input_hidden))
out = sigmoid(np.dot(hidden, weights_hidden_output))
predictions = out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))