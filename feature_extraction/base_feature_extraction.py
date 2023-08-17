from abc import abstractmethod

import numpy as np

class BaseFeatureExtraction:
    name = "base-feature-extraction"

    def __init__(self):
        self._feature_names = None

    def extract(self, texts):
        """Extract features from a list of texts.

        Arguments
        ---------
        texts: list
            List of texts.

        Returns
        -------
        numpy.ndarray
            Feature matrix.
        """
        # raise NotImplementedError
        self.fit(texts)
        return self.transform(texts)