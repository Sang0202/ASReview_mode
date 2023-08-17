from base_balance import BaseBalance

class SimpleBalance(BaseBalance):
    """Simple balance class that does not do anything."""

    name = "simple"
    label = "Simple (no balancing)"

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
        return X[train_idx], y[train_idx]