# models/tabnet_model.py

import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier

class TabNetRiskModel:
    def __init__(self, seed=42):
        self.model = TabNetClassifier(
            seed=seed,
            verbose=0
        )

    def fit(self, X_train, y_train, X_valid, y_valid, max_epochs=50, batch_size=256):
        self.model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_valid, y_valid)],
            eval_name=["valid"],
            eval_metric=["auc"],
            max_epochs=max_epochs,
            patience=10,
            batch_size=batch_size,
            virtual_batch_size=128
        )

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def feature_importances(self):
        return self.model.feature_importances_