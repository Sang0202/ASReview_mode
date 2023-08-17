from base_query import BaseQuery
import numpy as np

class QueryUncertain(BaseQuery):
    name = "query-uncertain"
    label = "Uncertainty Query"

    # def __init__(self):
    #     super().__init__()

    # def _query(self, prediction_proba, n_instances, X=None):
    #     query_idx = np.argsort(-prediction_proba[:, 1])[:n_instances]
    #     return query_idx
    
    def _query(self, prediction_proba, n_instances=1, X=None, threshold=None, labeled_idx=None):
        
        prediction_proba = prediction_proba*100 # convert to percentage


        top_2 = self.get_top_2(prediction_proba)
        diff_top_2 = self.get_diff_top_2(top_2)

        sorted_idx_d = np.argsort(-np.array(diff_top_2))[:len(labeled_idx)+n_instances]
        query_idx_d = np.array([elm for elm in sorted_idx_d if elm not in labeled_idx])[:n_instances]
        

        sorted_idx_u = np.argsort(np.array(diff_top_2))[:len(labeled_idx)+n_instances]
        query_idx_u = np.array([elm for elm in sorted_idx_u if elm not in labeled_idx])[:n_instances]
        
        return query_idx_d + query_idx_u

    