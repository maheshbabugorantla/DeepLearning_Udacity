import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

class NeuralNetwork(object):

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.hidden_nodes = hidden_nodes
        self.learning_rate = learning_rate

        # Initialize Weights
        # Weights between the Input and Hidden Layer
        # Generating the random samples from the gaussain distribution
        self.weights_input_to_hidden = np.random.normal(0, self.input_nodes**-0.5,
                                                            (self.input_nodes, self.hidden_nodes))

        # Weights between the Hidden and the Output Layer
        self.weights_hidden_to_output = np.random.normal(0, self.hidden_nodes**-0.5,
                                                            (self.hidden_nodes, self.output_nodes))

        self.activation_function = sigmoid

    # If the deriv is True this returns the derivative of the Sigmoid Function
    def sigmoid(x, deriv=False):

        if(deriv == True):
            return sigmoid(x) * (1 - sigmoid(x))

        return 1 / (1 + np.exp(-x))

    def train(self, features, targets):

        '''
            Train the network on a batch of features and targets

            @params
            features: This is 2D Array in which each row is one data record and each column is feature
            targets: This a 1D Array of Target Values
        '''
        n_records = features.shape[0] # No. of rows in the features array

        # Initializing the Delta Weights between each layer
        delta_weights_i_h = np.zeroes(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeroes(self.weights_hidden_to_output.shape)

        for x, y in zip(features, targets):
            # Implementing the forward pass here

            ### Implementing the Hidden Layer
            hidden_inputs = np.dot(x, self.weights_input_to_hidden) # Signals into the hidden Layer
            hidden_outputs = self.activation_function(hidden_inputs) # Signals from the hidden Layer using the activation function

            ### Implementing the Output Layer
            final_inputs = np.dot(hidden_outputs, weights_hidden_to_output)
            final_outputs = self.activation_function(final_inputs)

            ### Implementing the BackWard Pass

            ## Calculate the error between predicted output and valid output
            error = y - final_outputs
            output_error_term = error * self.activation_function(final_outputs, deriv=True)

            ### Calculating the Hidden Errors from each respective node from the Hidden Layer
            hidden_error = np.dot(weights_hidden_to_output, output_error_term)
            hidden_error_term = hidden_error * self.activation_function(hidden_outputs, deriv=True)

            delta_weights_i_h += hidden_error_term * x[:, None]
            delta_weights_h_o += output_error_term * hidden_outputs

        self.weights_input_to_hidden += self.learning_rate * (delta_weights_i_h / n_records)
        self.weights_hidden_to_output += self.learning_rate * (delta_weights_h_o / n_records)

    def run(self, features):

        '''
            Run a forward Pass through the network with the input features

            @params
            -------
            features: 1D array of feature values
        '''
        hidden_inputs = np.dot(features, self.weights_input_to_hidden) # Signals into the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

def get_hourly_dataset():
    return pd.read_csv("hour.csv")

def main():

    hourly_data = get_hourly_dataset()

    dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']

    for each in dummy_fields:
        dummies = pd.get_dummies(hourly_data[each], prefix=each, drop_first=False)
        hourly_data = pd.concat([hourly_data, dummies], axis=1)

    fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 'weekday',
                        'atemp', 'mnth', 'workingday', 'hr']

    data = hourly_data.drop(fields_to_drop, axis=1)

    quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']

    data = scale_target_variables(data, quant_features)

    # Getting the Test Data which will be used later on to perform the evaluation on the trained neural network
    test_data = data[-21*24:]

    # Removing the test Data from the Original Data
    data = data[:-21*24]

    # Dividing the data and test_data into features and target variables
    features, targets, test_features, test_targets = get_features_and_targets(data, test_data)

    # Now we will diving the features and targets data set into
    # Training and Validation Dataset
    features_training, targets_training = features[:-60*24], targets[:-60*24]
    features_validation, targets_validation = features[-60*24:], targets[-60*24:]



def get_features_and_targets(data, test_data):

    target_fields = ['cnt', 'casual', 'registered']
    features, targets = data.drop(target_fields, axis=1), data[target_fields]
    test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

    return (features, targets, test_features, test_targets)

def scale_target_variables(data, quant_features):

    scaled_features = dict()

    data = deepcopy(data)

    # Storing the scaling in a dictionary so that we can convert back later
    for each in quant_features:
        mean, std = data[each].mean(), data[each].std()
        scaled_features[each] = [mean, std]
        data.loc[:, each] = (data[each] - mean)/std

    return data

if __name__ == '__main__':
    main()
