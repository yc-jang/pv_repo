# moo_scatter_save.py
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ─────────────────────────── 공통 헬퍼 ───────────────────────────

def _split_cols(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """x::, f::, g:: 컬럼 목록을 분리해 반환한다."""
    x_cols = [c for c in df.columns if c.startswith("x::")]
    f_cols = [c for c in df.columns if c.startswith("f::")]
    g_cols = [c for c in df.columns if c.startswith("g::")]
    return x_cols, f_cols, g_cols


def _compute_cv(df: pd.DataFrame) -> np.ndarray:
    """제약 위반도(CV): g:: ≥ 0만 합산(양수=위반). g가 없으면 0."""
    g_cols = [c for c in df.columns if c.startswith("g::")]
    if not g_cols:
        return np.zeros(len(df), dtype=float)
    G = df[g_cols].to_numpy(float)
    return np.maximum(G, 0.0).sum(axis=1)


def _minmax_norm(a: np.ndarray) -> np.ndarray:
    """Min-Max 정규화(상수 → 0)."""
    a = np.asarray(a, dtype=float)
    mn, mx = np.nanmin(a), np.nanmax(a)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
        return np.zeros_like(a)
    return (a - mn) / (mx - mn)


def _knee_indices_by_utopia(df: pd.DataFrame, f_axes: Sequence[str], k: int = 3) -> np.ndarray:
    """Utopia(각 f 최소)로부터의 정규화 거리(min-norm) 기준으로 가장 짧은 k개 인덱스."""
    F = df[list(f_axes)].to_numpy(float)
    f_min, f_max = F.min(axis=0), F.max(axis=0)
    denom = np.where(f_max > f_min, f_max - f_min, 1.0)
    Fn = (F - f_min) / denom
    dist = np.linalg.norm(Fn, axis=1)
    k = max(1, min(k, len(dist)))
    return np.argsort(dist)[:k]


def _top_indices_by_sum(df: pd.DataFrame, f_axes: Sequence[str], top_n: int = 5) -> np.ndarray:
    """정규화 없이 f 합(작을수록 좋음) 기준 상위 top_n 인덱스."""
    s = df[list(f_axes)].to_numpy(float).sum(axis=1)
    return np.argsort(s)[: max(1, min(top_n, len(s)))]


def _ensure_dir(p: Path) -> None:
    """상위 디렉토리 생성."""
    p.parent.mkdir(parents=True, exist_ok=True)


def _safe_filename(name: str) -> str:
    """파일명 안전화."""
    return "".join(ch if ch.isalnum() or ch in "._- " else "_" for ch in name).strip()


# ─────────────────────────── Scatter Figure 생성 ───────────────────────────

def build_scatter_figure(
    pareto_df: pd.DataFrame,
    f_axes: Sequence[str],
    *,
    highlight_knee_k: int = 3,
    highlight_top_n: int = 5,
    title: Optional[str] = None,
) -> go.Figure:
    """2D/3D 목적공간 산점도 Figure 생성(무릎/Top 하이라이트 포함).

    Args:
        pareto_df: x::, f::, g:: 컬럼을 포함한 DataFrame.
        f_axes: 표시할 f 축 목록. 길이 2 → 2D, 길이 3 → 3D.
        highlight_knee_k: 무릎점 후보 개수.
        highlight_top_n: 상위해(합 최소) 개수.
        title: 레이아웃 타이틀(미지정 시 자동 생성).

    Returns:
        Plotly Figure.
    """
    # 축 검증
    if len(f_axes) not in (2, 3):
        raise ValueError("f_axes 길이는 2 또는 3이어야 합니다.")
    for c in f_axes:
        if c not in pareto_df.columns:
            raise KeyError(f"축 '{c}' 컬럼이 없습니다.")

    # 제약 분리
    cv = _compute_cv(pareto_df)
    feas_mask = (cv == 0.0)

    # 하이라이트 인덱스(지정 축 기준으로 계산)
    knee_idx = _knee_indices_by_utopia(pareto_df, f_axes, k=highlight_knee_k)
    top_idx = _top_indices_by_sum(pareto_df, f_axes, top_n=highlight_top_n)

    # 타이틀
    if title is None:
        title = ", ".join(f_axes) if len(f_axes) == 3 else f"{f_axes[0]} vs {f_axes[1]}"

    # 2D/3D 분기
    if len(f_axes) == 2:
        x, y = f_axes
        fig = go.Figure()
        # infeasible
        if np.any(~feas_mask):
            fig.add_trace(go.Scatter(
                x=pareto_df.loc[~feas_mask, x],
                y=pareto_df.loc[~feas_mask, y],
                mode="markers",
                name="infeasible",
                marker=dict(color="lightgray", size=6, opacity=0.6),
                hovertemplate=f"{x}: %{{x}}<br>{y}: %{{y}}<extra>infeasible</extra>",
            ))
        # feasible
        if np.any(feas_mask):
            fig.add_trace(go.Scatter(
                x=pareto_df.loc[feas_mask, x],
                y=pareto_df.loc[feas_mask, y],
                mode="markers",
                name="feasible",
                marker=dict(color="royalblue", size=7, opacity=0.85),
                hovertemplate=f"{x}: %{{x}}<br>{y}: %{{y}}<extra>feasible</extra>",
            ))
        # knee
        if knee_idx.size:
            fig.add_trace(go.Scatter(
                x=pareto_df.iloc[knee_idx][x],
                y=pareto_df.iloc[knee_idx][y],
                mode="markers",
                name="knee",
                marker=dict(symbol="star", color="crimson", size=12),
                hovertemplate=f"{x}: %{{x}}<br>{y}: %{{y}}<extra>knee</extra>",
            ))
        # top-n
        if top_idx.size:
            fig.add_trace(go.Scatter(
                x=pareto_df.iloc[top_idx][x],
                y=pareto_df.iloc[top_idx][y],
                mode="markers",
                name="top-n",
                marker=dict(symbol="triangle-up", color="seagreen", size=11),
                hovertemplate=f"{x}: %{{x}}<br>{y}: %{{y}}<extra>top-n</extra>",
            ))

        fig.update_layout(
            title=title, xaxis_title=x, yaxis_title=y,
            template="simple_white", hovermode="closest",
            legend=dict(itemclick="toggleothers"),
            margin=dict(l=40, r=20, t=40, b=40),
        )
        return fig

    else:
        x, y, z = f_axes
        fig = go.Figure()
        # infeasible
        if np.any(~feas_mask):
            fig.add_trace(go.Scatter3d(
                x=pareto_df.loc[~feas_mask, x],
                y=pareto_df.loc[~feas_mask, y],
                z=pareto_df.loc[~feas_mask, z],
                mode="markers",
                name="infeasible",
                marker=dict(size=3.5, color="lightgray", opacity=0.55),
            ))
        # feasible
        if np.any(feas_mask):
            fig.add_trace(go.Scatter3d(
                x=pareto_df.loc[feas_mask, x],
                y=pareto_df.loc[feas_mask, y],
                z=pareto_df.loc[feas_mask, z],
                mode="markers",
                name="feasible",
                marker=dict(size=4.5, color="royalblue", opacity=0.85),
            ))
        # knee
        if knee_idx.size:
            fig.add_trace(go.Scatter3d(
                x=pareto_df.iloc[knee_idx][x],
                y=pareto_df.iloc[knee_idx][y],
                z=pareto_df.iloc[knee_idx][z],
                mode="markers",
                name="knee",
                marker=dict(symbol="star", size=7.5, color="crimson"),
            ))
        # top-n
        if top_idx.size:
            fig.add_trace(go.Scatter3d(
                x=pareto_df.iloc[top_idx][x],
                y=pareto_df.iloc[top_idx][y],
                z=pareto_df.iloc[top_idx][z],
                mode="markers",
                name="top-n",
                marker=dict(symbol="triangle-up", size=7, color="seagreen"),
            ))

        fig.update_layout(
            title=f"3D: {x}, {y}, {z}",
            scene=dict(xaxis_title=x, yaxis_title=y, zaxis_title=z, aspectmode="cube"),
            template="simple_white",
            legend=dict(itemclick="toggleothers"),
            margin=dict(l=0, r=0, b=0, t=30),
        )
        return fig


# ─────────────────────────── 이미지 저장 API ───────────────────────────

def save_scatter_png(
    pareto_df: pd.DataFrame,
    f_axes: Sequence[str],
    out_dir: str | Path,
    *,
    filename: Optional[str] = None,
    highlight_knee_k: int = 3,
    highlight_top_n: int = 5,
    scale: float = 2.0,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Path:
    """2D/3D Scatter를 PNG로 저장한다.

    Args:
        pareto_df: x::, f::, g:: 컬럼을 포함한 DataFrame.
        f_axes: 표시할 f 축 목록. 길이 2 → 2D, 길이 3 → 3D.
        out_dir: 출력 디렉토리.
        filename: 파일명(확장자 제외). None이면 축 이름으로 자동 생성.
        highlight_knee_k: 무릎점 후보 개수.
        highlight_top_n: 상위해(합 최소) 개수.
        scale: 이미지 렌더 배율.
        width: 픽셀 폭(선택).
        height: 픽셀 높이(선택).

    Returns:
        저장된 PNG 경로.
    """
    fig = build_scatter_figure(
        pareto_df,
        f_axes,
        highlight_knee_k=highlight_knee_k,
        highlight_top_n=highlight_top_n,
    )

    # 파일명 구성
    if filename is None:
        if len(f_axes) == 2:
            filename = f"scatter_2d__{f_axes[0]}__{f_axes[1]}"
        else:
            filename = f"scatter_3d__{f_axes[0]}__{f_axes[1]}__{f_axes[2]}"
    fname = _safe_filename(filename) + ".png"
    out_path = Path(out_dir) / fname
    _ensure_dir(out_path)

    # 정적 이미지 저장(Plotly → kaleido 필요)
    try:
        fig.write_image(str(out_path), format="png", scale=scale, width=width, height=height)
    except ValueError as e:
        # kaleido 미설치 등 환경 문제를 명확히 안내
        raise RuntimeError(
            "이미지 저장에 실패했습니다. Plotly 정적 이미지 저장에는 'kaleido'가 필요합니다. "
            "가상환경에서 `pip install -U kaleido` 설치를 확인하세요."
        ) from e

    return out_path
