import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import fdsfunctions as fds
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

headers = ["satisfaction","Gender","Customer Type","Age","Type of Travel","Class","Flight Distance","Seat comfort","Departure/Arrival time convenient","Food and drink","Gate location","Inflight wifi service","Inflight entertainment","Online support","Ease of Online booking","On-board service","Leg room service","Baggage handling","Checkin service","Cleanliness","Online boarding","Departure Delay in Minutes","Arrival Delay in Minutes"]
data = pd.read_csv("./Invistico_Airline.csv", names=headers, header=0, engine='python')

# Cleanup the dataset
data.dropna(inplace=True)  # Remove all rows with missing values, they are only a few so the impact on the dataset is minimal
# Remove columns that have correlation with 'satisfaction' close to zero
data.drop(['Flight Distance','Departure/Arrival time convenient','Gate location','Departure Delay in Minutes','Arrival Delay in Minutes'], axis=1, inplace=True)

# Turn target into [0, 1]
Y = data.pop("satisfaction")
for i, v in Y.iteritems():
    if Y[i] == "satisfied":
        Y[i] = 1
    else:
        Y[i] = 0

# Encode qualitative columns
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0, 1, 3, 4])], remainder='passthrough')
data = np.array(ct.fit_transform(data), dtype=np.float)

# Rescale features into [0, 1] range for faster convergence
fds.rescale(data)

# Add intercept column
ones = np.ones((data.shape[0], 1))
data = np.append(ones, data, axis=1)

###### Run Gradient Ascent method ######
def Gradient_Ascent(data):
    # Split the dataset into 70% training, 30% testing
    X_train, X_test, Y_train, Y_test = train_test_split(data, Y, train_size=0.7, random_state=1)
    Y_train = np.array(Y_train, dtype=np.float32)
    Y_test = np.array(Y_test, dtype=np.float32)

    theta0 = np.zeros(X_train.shape[1])
    n_iter=1000

    theta_final, log_l_history, _ = fds.gradient_ascent(theta0, X_train, Y_train, fds.grad_l, alpha=0.3, iterations=n_iter)

    _, ax = plt.subplots(num=2)

    ax.set_ylabel('l(Theta)')
    ax.set_xlabel('Iterations')
    _ = ax.plot(range(len(log_l_history)), log_l_history, 'b.')
    plt.show()

    # Calculate accuracy
    correct_predictions = 0
    for i in range(X_test.shape[0]):
        prediction = theta_final.T.dot(X_test[i, :])
        if prediction < 0:
            prediction = 0
        else:
            prediction = 1
        if prediction == Y_test[i]:
            correct_predictions += 1

    # Obtain predictions using our trained model on testing data
    predictions = X_test.dot(theta_final)
    fds.plot_rpc(predictions, Y_test, True)

    print("Gradient Ascent Results:")    
    print(f"Number of correct predictions: {correct_predictions}, total predictions: {X_test.shape[0]}, accuracy: {correct_predictions / X_test.shape[0]}")
    return predictions, Y_test

GA_predictions, GA_Y_test = Gradient_Ascent(data)

###### Run Newton method ######
def Newton(data):
    # Train size needs to be small, otherwise matrix calculations take too long
    # also tends to create singular matrices, but this split works
    X_train, X_test, Y_train, Y_test = train_test_split(data, Y, train_size=0.01, random_state=1)
    Y_train = np.array(Y_train, dtype=np.float32)
    Y_test = np.array(Y_test, dtype=np.float32)

    theta0 = np.zeros(X_train.shape[1])

    theta_final, _, log_l_history = fds.newton(theta0, X_train, Y_train, fds.grad_l, fds.hess_l, 1e-6)

    _, ax = plt.subplots(num=2)

    ax.set_ylabel('l(Theta)')
    ax.set_xlabel('Iterations')
    _ = ax.plot(range(len(log_l_history)), log_l_history, 'b.')
    plt.show()

    # Calculate accuracy
    correct_predictions = 0
    for i in range(X_test.shape[0]):
        prediction = theta_final.T.dot(X_test[i, :])
        if prediction < 0:
            prediction = 0
        else:
            prediction = 1
        if prediction == Y_test[i]:
            correct_predictions += 1

    # Obtain predictions using our trained model on testing data
    predictions = X_test.dot(theta_final)
    fds.plot_rpc(predictions, Y_test, True)

    print("Newton Method Results:")    
    print(f"Number of correct predictions: {correct_predictions}, total predictions: {X_test.shape[0]}, accuracy: {correct_predictions / X_test.shape[0]}")
    return predictions, Y_test

Newton_predictions, Newton_Y_test = Newton(data)

###### Run Gaussian Discriminant Analysis method ######
def GDA(data):
    # Plot the distribution of all features to check if they're all normally distributed
    for column_idx in range(data.shape[1]):
        column = data[:, column_idx]
        plt.hist(column, alpha=1/5)
    plt.show()

    # Split the dataset into 70% training, 30% testing
    X_train, X_test, Y_train, Y_test = train_test_split(data, Y, train_size=0.7, random_state=1)
    Y_train = np.array(Y_train, dtype=np.float32)
    Y_test = np.array(Y_test, dtype=np.float32)

    # Train model
    predictor = LinearDiscriminantAnalysis()
    predictor.fit(X_train, Y_train)
    
    # Obtain predictions using our trained model on testing data
    predictions = predictor.predict(X_test)
    
    # Calculate accuracy
    correct_predictions = 0
    for i in range(len(predictions)):
        if predictions[i] == Y_test[i]:
            correct_predictions += 1

    fds.plot_rpc(predictions, Y_test, True)

    print("Gaussian Discriminant Analysis Results:")    
    print(f"Number of correct predictions: {correct_predictions}, total predictions: {X_test.shape[0]}, accuracy: {correct_predictions / X_test.shape[0]}")
    return predictions, Y_test

GDA_predictions, GDA_Y_test = GDA(data)

fds.plot_all_rpc([(GA_predictions, GA_Y_test, "GA"), (Newton_predictions, Newton_Y_test, "Newton"), (GDA_predictions, GDA_Y_test, "GDA")])
