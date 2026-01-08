from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def shap_beeswarm_plots(self, x_obj: pd.DataFrame, topk: int) -> tuple[pd.DataFrame, Any]:
    import shap  # SHAP==0.47.0
    import matplotlib.pyplot as plt
    from IPython.display import display

    cols = list(getattr(self, "columns", None) or x_obj.columns)
    X = x_obj.loc[:, cols]

    explainer = getattr(self, "explainer", None) or shap.TreeExplainer(self.model)
    self.explainer = explainer

    exp = explainer(X)
    if not isinstance(exp, shap.Explanation):
        exp = shap.Explanation(values=np.asarray(exp), data=X.to_numpy(), feature_names=cols)

    values = np.asarray(exp.values)  # (n_samples, n_features)

    # 1) 전역 랭킹(1..N): mean(|SHAP|)
    imp = np.abs(values).mean(axis=0)
    order = np.argsort(-imp)  # high -> low
    rank_to_feat = {r + 1: cols[i] for r, i in enumerate(order)}
    feat_to_rank = {cols[i]: r + 1 for r, i in enumerate(order)}

    important_set = set(getattr(self, "important_features", []))
    anchor_ranks = {feat_to_rank[f] for f in important_set if f in feat_to_rank}

    # 2) 선택 랭크 = (기본 topk 랭크) ∪ (important 랭크) 를 topk개로 맞춤
    selected = set(range(1, int(topk) + 1)).union(anchor_ranks)

    # important가 topk보다 많으면: important 중에서도 전역 중요도 상위(topk)만 유지
    if len(anchor_ranks) > topk:
        anchor_ranks = set(sorted(anchor_ranks)[: int(topk)])
        important_set = {rank_to_feat[r] for r in anchor_ranks}
        selected = set(anchor_ranks)

    # 초과되면 anchor(important) 아닌 것 중 덜 중요한(큰 rank)부터 제거
    if len(selected) > topk:
        removable = sorted([r for r in selected if r not in anchor_ranks], reverse=True)
        for r in removable:
            if len(selected) <= topk:
                break
            selected.remove(r)

    # 부족하면 topk 이후 랭크로 채움
    r = int(topk) + 1
    while len(selected) < topk:
        selected.add(r)
        r += 1

    selected = sorted(selected)  # 랭크 오름차순(1이 가장 중요)

    # 3) selected 내에서 important는 단독, 나머지 연속구간은 Other(a-b)로 묶기
    groups: list[tuple[str, list[str], tuple[int, int]]] = []
    i = 0
    while i < len(selected):
        rnk = selected[i]
        feat = rank_to_feat[rnk]

        if feat in important_set:
            groups.append((feat, [feat], (rnk, rnk)))
            i += 1
            continue

        start = i
        while i < len(selected):
            rr = selected[i]
            ff = rank_to_feat[rr]
            if ff in important_set:
                break
            i += 1

        rs = selected[start:i]
        members = [rank_to_feat[rr] for rr in rs]
        groups.append((f"Other ({rs[0]}-{rs[-1]})", members, (rs[0], rs[-1])))

    # 4) 그룹별 SHAP(샘플별 합)으로 beeswarm용 Explanation 생성
    idx = {c: j for j, c in enumerate(cols)}
    new_names = [name for name, _, _ in groups]

    new_vals = np.column_stack(
        [values[:, [idx[m] for m in members]].sum(axis=1) for _, members, _ in groups]
    )

    # 색상용 data: 단일 feature는 원본 값, Other는 NaN(회색)
    new_data = np.column_stack(
        [(X[members[0]].to_numpy() if len(members) == 1 else np.full(len(X), np.nan))
         for _, members, _ in groups]
    )

    new_exp = shap.Explanation(values=new_vals, data=new_data, feature_names=new_names)

    # 출력용 테이블(플롯은 +/- 그대로)
    df = pd.DataFrame(
        {
            "feature": new_names,
            "rank_range": [f"{a}-{b}" for _, _, (a, b) in groups],
            "members": [", ".join(members) for _, members, _ in groups],
            "mean_shap": new_vals.mean(axis=0),              # 방향(평균 부호) 참고용
            "mean_abs_shap": np.abs(new_vals).mean(axis=0),  # 크기 참고용
        }
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    plt.close("all")
    ax = shap.plots.beeswarm(new_exp, max_display=len(new_names), show=False)
    fig = ax.figure if hasattr(ax, "figure") else plt.gcf()

    display(df)
    display(fig)

    return df, fig


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
