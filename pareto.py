# viz_pymoo.py
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd

from pymoo.visualization.scatter import Scatter
from pymoo.visualization.pcp import PCP

# viz_moo_minimal.py  (pymoo + matplotlib만 사용)

from __future__ import annotations
from typing import Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd

from pymoo.visualization.scatter import Scatter
from pymoo.visualization.pcp import PCP


# ─────────────────────────── 공통 헬퍼 ───────────────────────────

def _split_cols(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    x_cols = [c for c in df.columns if c.startswith("x::")]
    f_cols = [c for c in df.columns if c.startswith("f::")]
    g_cols = [c for c in df.columns if c.startswith("g::")]
    return x_cols, f_cols, g_cols

def _compute_cv(df: pd.DataFrame) -> np.ndarray:
    g_cols = [c for c in df.columns if c.startswith("g::")]
    if not g_cols:
        return np.zeros(len(df), dtype=float)
    G = df[g_cols].to_numpy(float)
    return np.maximum(G, 0.0).sum(axis=1)

def _knee_indices_by_utopia(df: pd.DataFrame, f_cols: List[str], k: int = 3) -> np.ndarray:
    """Utopia(각 f 최소)로부터 정규화 거리 가장 짧은 k개 인덱스."""
    F = df[f_cols].to_numpy(float)
    f_min = F.min(axis=0)
    f_max = F.max(axis=0)
    denom = np.where(f_max > f_min, (f_max - f_min), 1.0)
    Fn = (F - f_min) / denom
    dist = np.linalg.norm(Fn, axis=1)
    k = max(1, min(k, len(dist)))
    return np.argsort(dist)[:k]

def _top_indices_by_sum(df: pd.DataFrame, f_cols: List[str], top_n: int = 5) -> np.ndarray:
    """f 합(작을수록 좋음) 기준 상위 top_n 인덱스."""
    s = df[f_cols].to_numpy(float).sum(axis=1)
    idx = np.argsort(s)[: max(1, min(top_n, len(s)))]
    return idx


# ───────────────────── 1) 목적공간 산점도(강화版) ─────────────────────
def plot_objective_scatter_pymoo(
    pareto_df: pd.DataFrame,
    f_pair: Optional[Tuple[str, str]] = None,
    highlight_knee_k: int = 3,
    highlight_top_n: int = 5,
) -> None:
    """f-공간 산점도: feasible/infesible 분리 + knee + topN 하이라이트."""
    _, f_cols, _ = _split_cols(pareto_df)
    if f_pair is None:
        f_cols = sorted(f_cols)
        if len(f_cols) < 2:
            raise ValueError("f:: 컬럼이 2개 이상 필요합니다.")
        f_pair = (f_cols[0], f_cols[1])

    # 데이터 준비
    F2 = pareto_df[[f_pair[0], f_pair[1]]].to_numpy(float)
    CV = _compute_cv(pareto_df)
    feas_mask = (CV == 0.0)

    # 하이라이트 후보 (전체 df 기준으로 뽑되, 표시 자체는 2D)
    knees = _knee_indices_by_utopia(pareto_df, [f_pair[0], f_pair[1]], k=highlight_knee_k)
    tops  = _top_indices_by_sum(pareto_df, [f_pair[0], f_pair[1]], top_n=highlight_top_n)

    # 플롯
    plot = Scatter(title=f"{f_pair[0]} vs {f_pair[1]}")

    # infeasible(회색), feasible(파랑)
    if np.any(~feas_mask):
        plot.add(F2[~feas_mask], color="lightgray", alpha=0.6, label="infeasible")
    if np.any(feas_mask):
        plot.add(F2[feas_mask], color="C0", alpha=0.8, label="feasible")

    # 하이라이트: knee(빨간 별), top-n(초록 삼각)
    if len(knees) > 0:
        plot.add(F2[knees], color="red", s=60, marker="*", label="knee")
    if len(tops) > 0:
        plot.add(F2[tops], color="green", s=50, marker="^", label="top-N (sum f)")

    plot.show()


# ───────────────────── 2) 목적들의 PCP(강화版) ─────────────────────
def plot_objectives_pcp_pymoo(
    pareto_df: pd.DataFrame,
    highlight_indices: Optional[Iterable[int]] = None,
) -> None:
    """모든 f:: 병렬좌표 + 선택 인덱스 강조(굵은 선)."""
    _, f_cols, _ = _split_cols(pareto_df)
    if not f_cols:
        raise ValueError("f:: 컬럼이 없습니다.")
    F = pareto_df[f_cols].to_numpy(float)

    # 기본선
    p = PCP(title="Objectives (PCP)", labels=f_cols)
    p.add(F, color="lightgray", alpha=0.6)

    # 강조선
    if highlight_indices:
        hi = np.array(list(highlight_indices), dtype=int)
        hi = hi[(hi >= 0) & (hi < len(pareto_df))]
        if len(hi) > 0:
            p.add(F[hi], linewidth=2.5, color="C3", alpha=0.9)

    p.show()


# ───────────────────── 3) 선택 X의 PCP(강화版) ─────────────────────
def plot_selected_x_pcp_pymoo(
    pareto_df: pd.DataFrame,
    x_cols: Iterable[str],
    highlight_indices: Optional[Iterable[int]] = None,
) -> None:
    """선택 x:: 축만 병렬좌표 + 선택 인덱스 강조(굵은 선)."""
    all_x, _, _ = _split_cols(pareto_df)
    sel = [c for c in x_cols if c in all_x]
    if not sel:
        raise ValueError("선택한 x:: 컬럼이 pareto_df에 없습니다.")

    X = pareto_df[sel].to_numpy(float)

    p = PCP(title="Selected X (PCP)", labels=sel)
    p.add(X, color="lightgray", alpha=0.6)

    if highlight_indices:
        hi = np.array(list(highlight_indices), dtype=int)
        hi = hi[(hi >= 0) & (hi < len(pareto_df))]
        if len(hi) > 0:
            p.add(X[hi], linewidth=2.5, color="C2", alpha=0.9)

    p.show()
