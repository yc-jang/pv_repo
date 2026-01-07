from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd


def shap_beeswarm_plots(self, topk: int = 20) -> Tuple[pd.DataFrame, Any]:
    """Render SHAP beeswarm (summary dot) plot and guarantee notebook display.

    Notes:
        - SHAP==0.47.0 기준: shap.plots.beeswarm 사용 (matplotlib backend).
        - display(fig)로 노트북 출력 보장.
        - beeswarm의 색상/분포를 위해 가능한 경우 X(data)를 Explanation에 포함.

    Args:
        topk: 표시할 상위 feature 개수.

    Returns:
        (df_summary, fig)
        - df_summary: feature별 mean_abs_shap 상위 topk 요약표
        - fig: matplotlib.figure.Figure (또는 유사 객체)
    """
    import shap  # SHAP 0.47.0
    import matplotlib.pyplot as plt
    from IPython.display import display

    shap_vals, x_obj = self._ensure_shap_values()

    # feature names
    if getattr(self, "columns", None) is None:
        n_feat = shap_vals.values.shape[1] if isinstance(shap_vals, shap.Explanation) else shap_vals.shape[1]
        cols = [f"f{i}" for i in range(n_feat)]
    else:
        cols = list(self.columns)

    # ensure Explanation for shap.plots.beeswarm
    if isinstance(shap_vals, shap.Explanation):
        exp = shap_vals
    else:
        X = x_obj if isinstance(x_obj, pd.DataFrame) else None
        data = X.to_numpy() if isinstance(X, pd.DataFrame) else None
        exp = shap.Explanation(
            values=np.asarray(shap_vals),
            data=data,
            feature_names=cols,
        )

    # summary table (mean |SHAP|)
    values = np.asarray(exp.values)
    imp = np.abs(values).mean(axis=0)
    df = (
        pd.DataFrame({"feature": cols, "mean_abs_shap": imp})
        .sort_values("mean_abs_shap", ascending=False)
        .head(int(topk))
        .reset_index(drop=True)
    )

    # beeswarm plot: matplotlib figure 확보 + display 보장
    plt.close("all")
    ax = shap.plots.beeswarm(exp, max_display=int(topk), show=False)
    fig = ax.figure if hasattr(ax, "figure") else plt.gcf()

    display(df)
    display(fig)

    return df, fig
