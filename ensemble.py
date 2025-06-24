import numpy as np
import shap
import pandas as pd
from sklearn.linear_model import LinearRegression

class EnsembleRunner:
    def __init__(self, model1, model2, meta_model=None):
        self.model1 = model1
        self.model2 = model2
        self.meta_model = meta_model or LinearRegression()
        self.is_meta_fitted = False

    def predict(self, X):
        pred1 = self.model1.model.predict(X)
        pred2 = self.model2.model.predict(X)

        stacked_preds = np.column_stack([pred1, pred2])

        if not self.is_meta_fitted:
            raise RuntimeError("Meta-model has not been trained. Call 'fit_meta_model' first.")

        ensemble_pred = self.meta_model.predict(stacked_preds)
        return ensemble_pred

    def fit_meta_model(self, X, y):
        pred1 = self.model1.model.predict(X)
        pred2 = self.model2.model.predict(X)

        stacked_preds = np.column_stack([pred1, pred2])

        self.meta_model.fit(stacked_preds, y)
        self.is_meta_fitted = True

    def get_shap_values(self, X, nsamples=100):
        explainer1 = shap.Explainer(self.model1.model)
        explainer2 = shap.Explainer(self.model2.model)

        shap_values1 = explainer1(X)
        shap_values2 = explainer2(X)

        weights = self.meta_model.coef_
        ensemble_shap_values = shap_values1.values * weights[0] + shap_values2.values * weights[1]

        return shap.Explanation(
            values=ensemble_shap_values,
            base_values=shap_values1.base_values * weights[0] + shap_values2.base_values * weights[1],
            data=X,
            feature_names=X.columns
        )

    def plot_shap_summary(self, X):
        shap_values = self.get_shap_values(X)
        shap.summary_plot(shap_values.values, X, feature_names=X.columns)

    def plot_shap_waterfall(self, X, index):
        shap_values = self.get_shap_values(X)
        shap.plots.waterfall(shap_values[index])
