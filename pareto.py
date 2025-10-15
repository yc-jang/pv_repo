# viz_pymoo.py
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd

from pymoo.visualization.scatter import Scatter
from pymoo.visualization.pcp import PCP


# ───────────────────────────── 공통 헬퍼 ─────────────────────────────

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


# ───────────────────────── 1) 목적공간 산점도 ─────────────────────────
def plot_objective_scatter_pymoo(pareto_df: pd.DataFrame,
                                 f_pair: Optional[Tuple[str, str]] = None) -> None:
    """f-공간 산점도 (pymoo Scatter)."""
    _, f_cols, _ = _split_cols(pareto_df)
    if not f_cols:
        raise ValueError("f:: 컬럼이 없습니다.")
    if f_pair is None:
        f_cols = sorted(f_cols)
        if len(f_cols) == 1:
            f_pair = (f_cols[0], f_cols[0])
        else:
            f_pair = (f_cols[0], f_cols[1])

    F = pareto_df[[f_pair[0], f_pair[1]]].to_numpy(float)
    Scatter(title=f"{f_pair[0]} vs {f_pair[1]}").add(F).show()


# ─────────────────────── 2) 목적 다차원 PCP (PCP) ───────────────────────
def plot_objectives_pcp_pymoo(pareto_df: pd.DataFrame) -> None:
    """모든 f:: 축을 병렬좌표로 표시 (pymoo PCP)."""
    _, f_cols, _ = _split_cols(pareto_df)
    if not f_cols:
        raise ValueError("f:: 컬럼이 없습니다.")
    F = pareto_df[f_cols].to_numpy(float)
    PCP(title="Objectives (PCP)", labels=f_cols).add(F).show()


# ─────────────────── 3) 선택 X의 PCP (최대 k개 축) ───────────────────
def plot_selected_x_pcp_pymoo(pareto_df: pd.DataFrame,
                              x_cols: Optional[Iterable[str]] = None,
                              k: int = 8) -> None:
    """선택한 x:: 축(최대 k개)만 병렬좌표로 표시."""
    all_x, _, _ = _split_cols(pareto_df)

    if x_cols:
        sel = [c for c in x_cols if c in all_x]
    else:
        # 분산 상위 k 자동 선택
        if not all_x:
            raise ValueError("x:: 컬럼이 없습니다.")
        var = pareto_df[all_x].var(numeric_only=True).sort_values(ascending=False)
        sel = var.index.tolist()
    sel = sel[:k]
    if not sel:
        raise ValueError("표시할 x:: 컬럼이 없습니다.")

    X = pareto_df[sel].to_numpy(float)
    PCP(title="Selected X (PCP)", labels=sel).add(X).show()
