import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

def linear_regression(filename):

    # Read Data
    dataframe = pd.read_fwf(filename)
    brain_weights = dataframe[['Brain']]
    body_weights = dataframe[['Body']]

    # Train the model on the data
    body_regression = linear_model.LinearRegression()
    body_regression.fit(brain_weights, body_weights)

    # Plot the results
    plt.scatter(brain_weights, body_weights) # This plots the original data points
    plt.plot(brain_weights, body_regression.predict(brain_weights)) # This plots the predicted data points
    plt.show()

def main():
    linear_regression('brain_body.txt')

if __name__ == '__main__':
    main()
