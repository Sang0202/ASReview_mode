from base_query import BaseQuery
import numpy as np

class QueryThreshold(BaseQuery):
    name = "query-threshold"
    label = "Threshold Query"

    # def __init__(self):
    #     super().__init__()

    # def _query(self, prediction_proba, n_instances, X=None):
    #     query_idx = np.argsort(-prediction_proba[:, 1])[:n_instances]
    #     return query_idx
    
    def _query(self, prediction_proba, n_instances=None, X=None, threshold=None, labeled_idx=None):
        
        prediction_proba = prediction_proba*100 # convert to percentage

        top_2 = self.get_top_2(prediction_proba)
        diff_top_2 = self.get_diff_top_2(top_2)

        if threshold is None:
            max_idx = np.argmin(diff_top_2[labeled_idx])
            original_idx = labeled_idx[max_idx]
            threshold = diff_top_2[original_idx]

        filtered_idx = np.where(diff_top_2 >= threshold)[0]
        sorted_idx = np.argsort(-np.array(diff_top_2[filtered_idx]))
        original_idx = filtered_idx[sorted_idx]
        query_idx = np.array([elm for elm in original_idx if elm not in labeled_idx])
        
        return query_idx