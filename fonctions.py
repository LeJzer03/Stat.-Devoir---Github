import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression, LogisticRegression

def get_normal_CI(data: np.array, confidence: float) -> tuple[float, float]:
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

    mean = np.mean(data)
    std_error = np.std(data, ddof=1) / np.sqrt(len(data))
    z_score = stats.norm.ppf(1 - (1 - confidence) / 2)
    lower_bound = mean - z_score * std_error
    upper_bound = mean + z_score * std_error
    return lower_bound, upper_bound


def get_bootstrap(data: np.array, confidence: float, n_resamples: int, fun=np.mean) -> tuple[float, float]:
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

    bootstrap_samples = [fun(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_resamples)]
    lower_bound = np.percentile(bootstrap_samples, (1 - confidence) / 2 * 100)
    upper_bound = np.percentile(bootstrap_samples, (1 + confidence) / 2 * 100)
    return lower_bound, upper_bound



def get_linear_model(data: np.array, y: np.array) -> tuple:
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

    # Initialize and fit the linear regression model
    model = LinearRegression()
    model.fit(data, y)

    # Make predictions on the training data
    y_pred = model.predict(data)

    return model, y_pred



def get_residue(y: np.array, y_pred: np.array) -> np.array:
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

    # Calculate the residuals
    residuals = y - y_pred

    return residuals



def get_logistic_model(data: np.array, y: np.array) -> tuple:
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
    # Initialize and fit the logistic regression model
    model = LogisticRegression()
    model.fit(data, y)

    # Make predictions on the training data
    y_pred = model.predict(data)

    return model, y_pred



def get_leverage(X: np.array) -> np.array:
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

    # Ajouter une colonne de biais (1) pour les équations normales
    X_bias = np.hstack([np.ones((X.shape[0], 1)), X])

    # Calcul de la matrice Hat (H = X(X'X)^(-1)X')
    H = X_bias @ np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T

    # Les leverage sont les diagonales de la matrice Hat
    leverage = np.diag(H)

    return leverage



def get_specific_residue_leverage(X: np.array, y: np.array, x_pos: np.array, y_pos: np.array) -> tuple[np.array, np.array]:
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

    residuals = []
    leverages = []

    for x, y_val in zip(x_pos, y_pos):
        # Ajouter le point à X et y
        X_extended = np.vstack([X, x])
        y_extended = np.append(y, y_val)

        # Ajuster le modèle de régression linéaire
        X_bias = np.hstack([np.ones((X_extended.shape[0], 1)), X_extended])
        beta = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y_extended

        # Calculer les prédictions et résidus
        y_pred = X_bias @ beta
        residual = y_extended - y_pred

        # Calculer la matrice Hat et les leverages
        H = X_bias @ np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T
        leverage = np.diag(H)

        # Stocker les résultats pour le nouveau point
        residuals.append(residual[-1])
        leverages.append(leverage[-1])

    return np.array(residuals), np.array(leverages)
