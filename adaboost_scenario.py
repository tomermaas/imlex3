import numpy as np
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from adaboost import AdaBoost
from decision_stump import DecisionStump
from loss_functions import misclassification_error

def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    model = AdaBoost(DecisionStump, n_learners)
    model.fit(train_X, train_y)

    # Initialize train and test errors with initial loss
    train_errors = [model.partial_loss(train_X, train_y, 1)]
    test_errors = [model.partial_loss(test_X, test_y, 1)]

    # Predict for all learners iteratively
    train_pred = np.zeros((train_size, n_learners))
    test_pred = np.zeros((test_size, n_learners))
    for t in range(n_learners):
        train_pred[:, t] = model.models_[t].predict(train_X) * model.weights_[t]
        test_pred[:, t] = model.models_[t].predict(test_X) * model.weights_[t]

    # Calculate errors incrementally for each T
    for T in range(2, n_learners + 1):
        train_errors.append(misclassification_error(train_y, np.sign(train_pred[:, :T].sum(axis=1))))
        test_errors.append(misclassification_error(test_y, np.sign(test_pred[:, :T].sum(axis=1))))

    # Plot train and test errors
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_errors, mode='lines', name='Train Error'))
    fig.add_trace(go.Scatter(y=test_errors, mode='lines', name='Test Error'))
    fig.update_layout(
        title='Train and Test Errors vs. Number of Learners',
        xaxis_title='Number of Learners',
        yaxis_title='Error'
    )
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    raise NotImplementedError()

    # Question 3: Decision surface of best performing ensemble
    raise NotImplementedError()

    # Question 4: Decision surface with weighted samples
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0.0)