import logging

from sklearn.linear_model import LinearRegression
from abc import ABC, abstractmethod


class Model(ABC):
    """
    Abstract class for all models
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model

        Args:
            X_train: Training data
            y_train: Training labels
        Returns:
            None
        """
        pass


class LinearRegressionModel(Model):
    """
    Linear Regression Model
    """

    def train(self, X_train, y_train, **kwargs):
        """
        Trains the model

        Args:
            X_train: Training Data
            y_train: Training Labels
        Retuns:
            None
        """
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model Training Completed")
            return reg
        except Exception as e:
            logging.error("Error in training model: {}".format(e))
            raise e
