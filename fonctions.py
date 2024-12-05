import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression, LogisticRegression

def get_normal_CI(data: np.array, confidence:float)->tuple[float, float]:
    """
    Computes an approximate normal confidence interval for the mean of the given data.

    Args:
    -----
        - `data` (np.array): A 1-dimensional array containing the data for which the confidence interval is to be calculated.
        - `confidence` (float): Confidence level, representing 1-alpha (e.g., 0.95 for a 95% confidence interval).

    Returns:
    --------
        tuple[float, float]: A tuple containing the lower and upper bounds of the confidence interval.
    """
    
    return None


def get_bootstrap(data: np.array, confidence:float, n_resamples:int, fun = np.mean)->tuple[float, float]:
    """
    Computes a confidence interval estimation using the bootstrap method (percentile method).

    Args:
    -----
        - `data` (np.array): A 1-dimensional array containing the data for which the confidence interval is to be calculated.
        - `confidence` (float): Confidence level, representing 1-alpha (e.g., 0.95 for a 95% confidence interval).
        - `n_resamples` (int): The number of bootstrap resamples to generate.
        - `fun` (callable, optional): A function applied to each resample to calculate the statistic of interest. Defaults to `np.mean`.

    Returns:
    --------
        tuple[float, float]: A tuple containing the lower and upper bounds of the confidence interval.
    """


    return None


def get_linear_model(data: np.array, y: np.array)->tuple:
    """
    Fits a linear regression model using the provided data and observed values, 
    and makes predictions on the training data.

    Args:
    -----
        - `data` (np.array): A 2-dimensional array containing the predictive variables (features).
        - `y` (np.array): A 1-dimensional array containing the observed values (target variable).

    Returns:
    --------
        tuple:
            - sklearn.linear_model.LinearRegression: The fitted linear regression model.
            - np.array: Predictions made by the model on the training data.

    """

    return None


def get_residue(y: np.array, y_pred: np.array)->np.array:
    """
    Computes the residuals (differences) between observed values and model predictions.

    Args:
    -----
        - `y` (np.array): A 1-dimensional array of observed values (ground truth).
        - `y_pred` (np.array): A 1-dimensional array of predicted values from a model.

    Returns:
    --------
        np.array: A 1-dimensional array containing the residuals.
    """

    return None


def get_logistic_model(data: np.array, y: np.array)->tuple:
    """
    Fits a logistic regression model using the provided data and observed values, 
    and makes predictions on the training data.

    Args:
    -----
        - `data` (np.array): A 2-dimensional array containing the predictive variables (features).
        - `y` (np.array): A 1-dimensional array containing the observed values (target variable).

    Returns:
    --------
        tuple:
            - sklearn.linear_model.LogisticRegression: The fitted logistic regression model.
            - np.array: Predictions made by the model on the training data.
    """

    return None


def get_leverage(X: np.array)->np.array:
    """
    Computes the leverage for each crystallisation of the predictive variables.

    Args:
    -----
        - `X` (np.array): A 2-dimensional array representing the matrix of crystallisations 
                    of the predictive variables (features).

    Returns:
    --------
        np.array: A 1-dimensional array containing the leverage values for each crystallisation.
    """

    return None


def get_specific_residue_leverage(X: np.array, y:np.array, x_pos:np.array, y_pos:np.array)->tuple[np.array, np.array]:
    """
    Computes the residuals and leverage for a group of specific cristallisations to be added to 
    the initial dataset.

    Args:
    -----
        X (np.array): A 2-dimensional array representing the initial matrix of 
                      crystallisations of the predictive variables (features).
        y (np.array): A 1-dimensional array of the initial observed variables (target values).
        x_pos (np.array): A 1-dimensional array of predictive variable values to be added to `X` (only the features, no bias in the argument).
        y_pos (np.array): A 1-dimensional array of observed variable values to be added to `y`.

    Returns:
    --------
        tuple[np.array, np.array]:
            - np.array: Residuals for each position specified by `x_pos` and `y_pos`.
            - np.array: Leverage values for each position specified by `x_pos` and `y_pos`.
    """

    return None