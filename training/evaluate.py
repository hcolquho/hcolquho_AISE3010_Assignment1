# Module to compute accuracy for classification and MSE for regression
import numpy as np

def accuracy(model, X, y):
    y_pred = model.forward(X) # get model predictions
    y_pred_classes = (y_pred >= 0.5).astype(int) # threshold at 0.5 for binary classification
    correct_predictions = np.sum(y_pred_classes == y) # count correct predictions
    accuracy = correct_predictions / y.shape[0] # compute accuracy
    return accuracy

def mse(model, X, y):
    y_pred = model.forward(X) # get model predictions
    mse = np.mean((y - y_pred) ** 2) # compute Mean Squared Error
    return mse