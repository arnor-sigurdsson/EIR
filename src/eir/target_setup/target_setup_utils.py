from sklearn.base import BaseEstimator, TransformerMixin


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X
