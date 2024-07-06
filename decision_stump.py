from __future__ import annotations
from typing import Tuple, NoReturn
from base_estimator import BaseEstimator
import numpy as np
from itertools import product
from loss_functions import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a decision stump to the given data. That is, finds the best feature and threshold by which to split

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        n_samples, n_features = X.shape

        best_err, best_j, best_sign, best_thr = np.inf, None, None, None
        for j, sign in product(range(n_features), [-1, 1]):
            # Find best threshold for feature j
            X_j = X[:, j]
            thr, thr_err = self._find_threshold(X_j, y, sign)

            if thr_err < best_err:
                best_err, best_j, best_sign, best_thr = thr_err, j, sign, thr

        # Store best parameters
        self.threshold_ = best_thr
        self.j_ = best_j
        self.sign_ = best_sign

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict sign responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return self.sign_ * np.sign(X[:, self.j_] - self.threshold_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """

        # Sort the values and labels TOGETHER
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_labels = labels[sorted_indices]

        n_samples = len(labels)
        n_pos = np.sum(sorted_labels == 1)  # Count of positive labels
        n_neg = n_samples - n_pos  # Count of negative labels

        err_left = n_pos  # Initially, all positive samples are on the left
        err_right = n_neg  # Initially, all negative samples are on the right

        best_thr, best_err = sorted_values[0] - 1, 1.0  # Initialize with worst-case error

        # Check if all weights are the same
        if np.all(sorted_labels == sorted_labels[0]):
            return sorted_values[0], 0.5  # Return the first threshold with error 0.5

        for i in range(n_samples - 1):
            # Update error counts based on the current label
            if sorted_labels[i] == 1:
                err_left -= 1
                err_right += 1
            else:
                err_left += 1
                err_right -= 1

            # Calculate the current misclassification error on both sides
            curr_err_left = err_left / n_samples
            curr_err_right = err_right / n_samples

            # Update best threshold and error if we find a better one
            if curr_err_left < best_err and sorted_values[i] != sorted_values[i + 1]:
                best_err = curr_err_left
                best_thr = sorted_values[i]

            if curr_err_right < best_err and sorted_values[i] != sorted_values[i + 1]:
                best_err = curr_err_right
                best_thr = sorted_values[i + 1]

        return best_thr, best_err


    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_pred = self._predict(X)
        return misclassification_error(y, y_pred)
