from sklearn.base import BaseEstimator, TransformerMixin
import miceforest as mf


# noinspection PyUnusedLocal
class MiceForest(BaseEstimator, TransformerMixin):
    def __init__(self, iterations=1, random_state=None):
        self.iterations = iterations
        self.random_state = random_state
        self.kds = None

    def fit(self, X, y=None):
        self.kds = mf.ImputationKernel(
            X,
            random_state=self.random_state,
            data_subset=0.5,
            train_nonmissing=True,
            mean_match_candidates={
                24: 0,
                26: 0
            },
        )
        self.kds.mice(
            self.iterations,
            n_estimators=100,
        )
        return self

    def transform(self, X, y=None):
        return self.kds.impute_new_data(X).complete_data(0)
