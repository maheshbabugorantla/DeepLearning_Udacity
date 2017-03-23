import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from numpy import sqrt
import matplotlib.pyplot as mp

def linear_regression_challenge(filename):

    # Read the Challenge Dataset
    dataframe = pd.read_csv(filename, usecols=[0, 1], header=None, names=["Brain", "Body"]) # Adding the header to the file as there is no header
    brain_weights = dataframe[["Brain"]]
    body_weights = dataframe[["Body"]]

    # Training the model with Linear Regression
    body_regression = linear_model.LinearRegression()
    body_regression.fit(brain_weights, body_weights)

    # Plotting the values
    predicted_body_weights = body_regression.predict(brain_weights)
    mp.scatter(brain_weights, body_weights) # Plotting the original points
    mp.plot(brain_weights, predicted_body_weights) # Plotting the predicted values
    mp.show()

    # Calculating the Error
    rms_error = sqrt(mean_squared_error(body_weights, predicted_body_weights))
    print("Error is " + str(rms_error))

def main():

    linear_regression_challenge('challenge_dataset.txt')

if __name__ == '__main__':
    main()
