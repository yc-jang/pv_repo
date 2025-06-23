import shap
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from typing import Dict

class XGBoostRunner:
    def __init__(self, model: XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.explainer = shap.Explainer(model)
        self.shap_values = self.explainer(self.X_test)
        self.data_mean = self.X_test.mean()

    def plot_beeswarm(self) -> widgets.Output:
        out = widgets.Output()
        with out:
            shap.plots.beeswarm(self.shap_values, max_display=20)
            plt.show()
        return out

    def plot_waterfall_with_mean(self, input_data: Dict[str, float]) -> widgets.Output:
        out = widgets.Output()
        input_df = pd.DataFrame([input_data], columns=self.X_test.columns)
        input_shap_values = self.explainer(input_df)

        # feature_names에 mean 추가
        mean_added_feature_names = [
            f"{col}\n(mean={self.data_mean[col]:.2f})"
            for col in self.X_test.columns
        ]

        # Waterfall Plot
        with out:
            shap.plots.waterfall(
                input_shap_values[0],
                max_display=20,
                feature_names=mean_added_feature_names
            )
            plt.show()
        return out

    def provide_recommendations(self, input_data: Dict[str, float]) -> widgets.Output:
        out = widgets.Output()
        input_df = pd.DataFrame([input_data], columns=self.X_test.columns)
        input_shap_values = self.explainer(input_df)

        recommendations = []
        for feature in self.X_test.columns:
            input_val = input_df.at[0, feature]
            mean_val = self.data_mean[feature]
            shap_val = input_shap_values.values[0, self.X_test.columns.get_loc(feature)]

            # 개선된 추천 로직:
            # SHAP 절대값이 클수록 중요하고, mean과의 차이가 클수록 강하게 조언
            importance = abs(shap_val)
            mean_diff = input_val - mean_val

            if shap_val > 0:
                direction = "↓ Decrease" if mean_diff > 0 else "↑ Increase"
            else:
                direction = "↑ Increase" if mean_diff < 0 else "↓ Decrease"

            strength = "Moderate"
            if importance > np.percentile(abs(self.shap_values.values), 75):
                strength = "Strong"
            elif importance < np.percentile(abs(self.shap_values.values), 25):
                strength = "Weak"

            recommendations.append({
                "Feature": feature,
                "Current": input_val,
                "Mean": mean_val,
                "SHAP Value": shap_val,
                "Suggested Action": direction,
                "Recommendation Strength": strength
            })

        rec_df = pd.DataFrame(recommendations).sort_values(
            by=["Recommendation Strength", "SHAP Value"], ascending=[False, False]
        )

        with out:
            display(rec_df.style.set_caption("📌 Optimized Recommendations"))
        return out

    def full_analysis_widget(self, input_data: Dict[str, float]) -> widgets.VBox:
        beeswarm_widget = self.plot_beeswarm()
        waterfall_widget = self.plot_waterfall_with_mean(input_data)
        recommendations_widget = self.provide_recommendations(input_data)

        return widgets.VBox([
            widgets.HTML("<h3>1️⃣ SHAP Beeswarm Plot (Global Explanation)</h3>"),
            beeswarm_widget,
            widgets.HTML("<h3>2️⃣ SHAP Waterfall Plot (Your Input)</h3>"),
            waterfall_widget,
            widgets.HTML("<h3>3️⃣ Actionable Recommendations</h3>"),
            recommendations_widget
        ])
