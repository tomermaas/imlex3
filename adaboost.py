import numpy as np
from typing import Callable, NoReturn
from base_estimator import BaseEstimator
from loss_functions import misclassification_error
from decision_stump import DecisionStump

class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations

    self.weights_: List[float]
        List of weights for each fitted estimator, fitted along the boosting iterations

    self.D_: List[np.ndarray]
        List of weights for each sample, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        n_samples = len(y)
        D = np.full(n_samples, 1 / n_samples)
        self.models_, self.weights_, self.D_ = [], [], []

        # Create a single DecisionStump instance
        model = self.wl_()

        for t in range(self.iterations_):
            self.D_.append(D.copy())

            # Re-initialize the DecisionStump parameters
            model.threshold_ = None
            model.j_ = None
            model.sign_ = None

            # Train the same weak learner with the updated weights
            model.fit(X, y)
            y_pred = model.predict(X)

            # Calculate error and alpha - WITH CORRECTION FOR ZERO ERROR
            epsilon = np.sum(D[y_pred != y])
            err = epsilon / np.sum(D)
            if err == 0:
                err = 1e-10

            alpha = 0.5 * np.log((1 - err) / err)

            # Update sample weights
            D *= np.exp(-alpha * y * y_pred)
            D /= np.sum(D)  # Normalize weights

            # Store model and weight
            self.models_.append(model)  # This will append a reference to the same model object
            self.weights_.append(alpha)



    def _predict(self, X):
        """
        Predict responses for given samples using fitted estimator over all boosting iterations

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return np.sign(np.sum([alpha * model.predict(X) for alpha, model in zip(self.weights_, self.models_)], axis=0))

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function over all boosting iterations

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
        y_pred = self.predict(X)
        return misclassification_error(y, y_pred)


    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators up to T learners

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return np.sign(np.sum([alpha * model.predict(X) for alpha, model in zip(self.weights_[:T], self.models_[:T])], axis=0))


    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function using fitted estimators up to T learners

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_pred = self.partial_predict(X, T)
        return self._loss(X, y)  # Use _loss function for consistency
