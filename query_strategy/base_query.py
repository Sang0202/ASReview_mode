from abc import abstractmethod
import numpy as np

class BaseQueryStrategy:
    '''Base class for query strategies.'''
    name = 'base-query-strategy'

    def query(self, X, classifier, n_instance=None):
        '''Query new instances.

        Arguments
        ---------
        X: numpy.ndarray
            Feature matrix.
        classifier: sklearn.base.BaseEstimator
            Classifier.
        n_instance: int
            Number of instances to query.

        Returns
        -------
        numpy.ndarray
            Indices of instances to query.
        '''
        if n_instance is None:
            n_instance = X.shape[0]
        
        prediction_proba = classifier.predict_proba(X)

        query_idx = self._query(prediction_proba, n_instance, X)

        return query_idx
    
    def get_top_2(self, prediction_proba):
        top_2 = []
        for record_proba in prediction_proba:
            idx_top_2 = np.argsort(record_proba)[:2]
            top_2.append(np.argsort(record_proba)[:2])

        return top_2
    
    def get_diff_top_2(self, array_top_2):
        diff_top_2 = []
        for record_proba in array_top_2:
            diff_top_2.append(record_proba[1] - record_proba[0])
        return diff_top_2
    
    def get_quotient_top_2(self, array_top_2):
        quotient_top_2 = []
        for record_proba in array_top_2:
            quotient_top_2.append(record_proba[1] / record_proba[0])
        return quotient_top_2
    
    @abstractmethod
    def _query(self, prediction_proba, n_instances, X=None):
        raise NotImplementedError