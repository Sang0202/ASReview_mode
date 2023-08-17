from base_feature_extraction import BaseFeatureExtraction
from sklearn.feature_extraction.text import TfidfVectorizer

class TfidfFeatureExtraction(BaseFeatureExtraction):
    name = "tfidf"
    label = "TF-IDF"

    def __init__(self, stop_words='english', ngram_max=1):
        super().__init__()
        # self._vectorizer = TfidfVectorizer()
        self.stop_words = stop_words
        self.ngram_max = ngram_max
        if stop_words is None or stop_words.lower() == 'none':
            sklearn_stop_words = None
        else:
            sklearn_stop_words = stop_words
        self._model = TfidfVectorizer(stop_words=sklearn_stop_words, ngram_range=(1, ngram_max))

    def fit(self, texts):
        """Fit the feature extraction model.

        Arguments
        ---------
        texts: list
            List of texts.
        """
        self._model.fit(texts)

    def transform(self, texts):
        """Transform the texts into a feature matrix.

        Arguments
        ---------
        texts: list
            List of texts.

        Returns
        -------
        numpy.ndarray
            Feature matrix.
        """
        return self._model.transform(texts).toarray()
    

    # def extract(self, texts):
    #     """Extract features from a list of texts.

    #     Arguments
    #     ---------
    #     texts: list
    #         List of texts.

    #     Returns
    #     -------
    #     numpy.ndarray
    #         Feature matrix.
    #     """
    #     return self._vectorizer.fit_transform(texts).toarray()