#define a class for base review
from abc import ABC, abstractmethod
from classifier_nb import NaiveBayesClassifier
from query_mix import MixQueryStrategy
from balance_simple import SimpleBalance
from feature_tfidf import TfidfFeatureExtraction

class BaseReview(ABC):
    name = "base-review"

    def __init__(
        self,
        input_data, #input_data has 3 columns:"id", "text" and "label"
        model=NaiveBayesClassifier(),
        query_model=MixQueryStrategy(),
        balance_model=SimpleBalance(),
        feature_model=TfidfFeatureExtraction(),
        n_instances=5,
        #labeled=[], #labeled is a list of indices of labeled papers
    ):
        super().__init__()

        self.classifier = model
        self.query_model = query_model
        self.balance_model = balance_model
        self.feature_model = feature_model

        self.input_data = input_data
        self.n_instances = n_instances
        self.labeled = input_data[["id", "label"]]

        if self.labeled is None:
            self.labeled = np.full(len(input_data), False, dtype=bool)
            self._label_priors()

        self.record_table = input_data[["id","text"]]
        self.X = self.feature_model.fit_transform(input_data["text"])

    def review(self):
        
        i = 0
        while i < len(self.input_data) - len(self.labeled):
            #Train a new model based on the labeled data
            self.train()

            #Query new records to label/review
            query_idx = self.query_model.query(self.X, self.classifier, n_instances=self.n_instances, labeled_idx=self.labeled.index.values) 

            #Label the new records
            labels = self._label(query_idx)

    def _label_priors(self):
        """Label priors.

        Returns
        -------
        numpy.ndarray
            Label priors.
        """
        #Get indices of unlabelled records (i.e. records that have not been reviewed)
        unlabeled_idx = np.where(~self.labeled)[0]
        self._label(unlabeled_idx[:self.n_instances]) 

    # def _query(self, n):
    #     """Query new instances.

    #     Arguments
    #     ---------
    #     n: int
    #         Number of instances to query.

    #     Returns
    #     -------
    #     numpy.ndarray
    #         Indices of instances to query.
    #     """
    #     #Get indices of unlabelled records (i.e. records that have not been reviewed)
    #     # unlabeled_idx = np.where(~self.labeled)[0]
    #     return self.query_model.query(self.X, self.classifier, n_instances=n, labeled_idx=self.labeled.index.values)
    
    def _label(self, record_idx):
        """Label new instances.

        Arguments
        ---------
        record_idx: numpy.ndarray
            Indices of instances to label.

        Returns
        -------
        numpy.ndarray
            Labels.
        """
        #Get indices of unlabelled records (i.e. records that have not been reviewed)
        # unlabeled_idx = np.where(~self.labeled)[0]
        
        pass

    def train(self):
        """Train a new model based on the labeled data."""
        
        y_sample_input = (
            self.record_table
            .merge(self.labeled, how='left', on='id')
            .loc[:, "label"]
            .fillna(-1)
            .to_numpy()
        )

        train_idx = np.where(y_sample_input != -1)[0]

        X_train, y_train = self.balance_model.sample(self.X, y_sample_input, train_idx)

        self.classifier.fit(X_train, y_train)

        