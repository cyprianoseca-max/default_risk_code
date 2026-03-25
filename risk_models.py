# models/risk_models.py

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class LRModel:
    def __init__(self, max_iter=1000):
        self.model = LogisticRegression(max_iter=max_iter)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


class RFModel:
    def __init__(self, n_estimators=200, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]