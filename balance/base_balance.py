from abc import abstractmethod

class BaseBalance:
    name = "base-balance"

    @abstractmethod
    def sample(self, X, y, train_idx):
        """Resample the training data.

        Arguments
        ---------
        X: numpy.ndarray
            Complete feature matrix.
        y: numpy.ndarray
            Labels for all papers.
        train_idx: numpy.ndarray
            Training indices, that is all papers that have been reviewed.

        Returns
        -------
        numpy.ndarray, numpy.ndarray
            X_train, y_train: the resampled matrix, labels.
        """
        raise NotImplementedError