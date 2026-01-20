from __future__ import annotations

from typing import Tuple
from pathlib import Path
import importlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ----------------------------
# Core functions
# ----------------------------
def _sort_lot_df(df: pd.DataFrame) -> pd.DataFrame:
    """LOT 정렬: lot_date_str -> lot_seq."""
    return df.sort_values(["lot_date_str", "lot_seq"]).reset_index(drop=True)


def _month_label(s: pd.Series) -> pd.Series:
    """lot_date_str(YYYYMMDD...) -> YYYY-MM."""
    dt = pd.to_datetime(s.astype(str).str.slice(0, 8), format="%Y%m%d", errors="coerce")
    return dt.dt.strftime("%Y-%m")


def _top5pct_mask(df: pd.DataFrame, residual_col: str = "residual") -> pd.Series:
    """|residual| 상위 5% 마스크."""
    abs_err = df[residual_col].abs()
    thr = abs_err.quantile(0.95)
    return abs_err >= thr


def _add_top5_month_bands(
    fig: go.Figure,
    df_sorted: pd.DataFrame,
    x_col: str = "index",
    lot_date_col: str = "lot_date_str",
    residual_col: str = "residual",
    band_color: str = "rgba(0,200,0,0.15)",
) -> None:
    """Top5% 오차가 존재하는 '월' 구간 band + 우측 상단 월 라벨."""
    d = df_sorted.copy()
    d["_month"] = _month_label(d[lot_date_col])
    d["_is_top5"] = _top5pct_mask(d, residual_col=residual_col)

    spans = (
        d.loc[d["_is_top5"], ["_month", x_col]]
        .groupby("_month", as_index=False)
        .agg(x0=(x_col, "first"), x1=(x_col, "last"))
        .dropna()
    )

    for _, row in spans.iterrows():
        x0, x1, m = row["x0"], row["x1"], row["_month"]

        fig.add_shape(
            type="rect",
            xref="x",
            yref="paper",
            x0=x0,
            x1=x1,
            y0=0,
            y1=1,
            fillcolor=band_color,
            line=dict(width=0),
            layer="below",
        )

        fig.add_annotation(
            x=x1,
            y=1,
            xref="x",
            yref="paper",
            text=str(m),
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(color="black"),
        )


def make_line_fig_with_bands(
    df: pd.DataFrame,
    title: str = "Actual vs Predicted (LOT ordered)",
) -> go.Figure:
    """Line plot + Top5% 오차 월 band."""
    d = _sort_lot_df(df)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["index"], y=d["y_true"], mode="lines+markers", name="Actual"))
    fig.add_trace(go.Scatter(x=d["index"], y=d["y_pred"], mode="lines+markers", name="Predicted"))

    fig.update_layout(
        title=title,
        xaxis_title="LOT",
        yaxis_title="Value",
        margin=dict(l=50, r=30, t=60, b=80),
        legend_title="Series",
    )
    fig.update_xaxes(type="category")

    _add_top5_month_bands(fig, d)
    return fig


def make_pred_vs_true_fig(
    df: pd.DataFrame,
    title: str = "Pred vs True (y = x)",
) -> go.Figure:
    """y_true vs y_pred 산점도 + y=x."""
    d = df.dropna(subset=["y_true", "y_pred"]).copy()
    x = d["y_true"].to_numpy()
    y = d["y_pred"].to_numpy()

    mn = float(np.min([x.min(), y.min()]))
    mx = float(np.max([x.max(), y.max()]))
    pad = (mx - mn) * 0.03 if mx > mn else 1.0
    x0, x1 = mn - pad, mx + pad

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d["y_true"],
            y=d["y_pred"],
            mode="markers",
            name="Points",
            marker=dict(size=7, opacity=0.85),
            customdata=np.stack([d["index"].astype(str).to_numpy()], axis=1),
            hovertemplate="LOT: %{customdata[0]}<br>y_true: %{x:.4f}<br>y_pred: %{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(go.Scatter(x=[x0, x1], y=[x0, x1], mode="lines", name="y = x"))

    fig.update_layout(
        title=title,
        xaxis_title="y_true",
        yaxis_title="y_pred",
        margin=dict(l=60, r=30, t=60, b=60),
    )
    fig.update_xaxes(range=[x0, x1])
    fig.update_yaxes(range=[x0, x1], scaleanchor="x", scaleratio=1)
    return fig


def build_figures(df: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    """최종 2개 Figure 반환."""
    return (make_line_fig_with_bands(df), make_pred_vs_true_fig(df))


# ----------------------------
# Verification / export (sample)
# ----------------------------
def _ensure_plotly_kaleido_ready() -> None:
    """
    Plotly(5.3.x) + kaleido 조합에서 write_image가 바로 안될 수 있어
    plotly.io._kaleido reload로 엔진 인식을 안정화.
    """
    import kaleido  # noqa: F401
    import plotly.io._kaleido as _k
    importlib.reload(_k)


def verify_and_export_sample_images(out_dir: Path) -> Tuple[Path, Path]:
    """샘플 데이터로 2개 figure 생성 후 PNG 저장."""
    _ensure_plotly_kaleido_ready()

    np.random.seed(7)

    dates = pd.date_range("2025-06-01", "2025-09-30", freq="D")
    rows = []
    for dt in dates:
        n = np.random.randint(3, 8)
        for seq in range(1, n + 1):
            lot_date_str = dt.strftime("%Y%m%d")
            line_letter = np.random.choice(list("ABM"))
            lot = f"N87{line_letter}{lot_date_str}{seq:02d}"
            rows.append((lot, lot_date_str, seq, line_letter))

    df = pd.DataFrame(rows, columns=["index", "lot_date_str", "lot_seq", "line"])

    base = 100 + np.sin(np.linspace(0, 8 * np.pi, len(df))) * 2
    df["y_true"] = base + np.random.normal(0, 0.6, len(df))
    df["y_pred"] = df["y_true"] + np.random.normal(0, 0.8, len(df))

    # 8월에 큰 오차를 몰아 넣어 band가 눈에 잘 보이게
    is_aug = df["lot_date_str"].str.startswith("202508")
    idx_aug = np.where(is_aug)[0]
    out_idx = np.random.choice(idx_aug, size=max(1, int(len(df) * 0.05)), replace=False)
    df.loc[out_idx, "y_pred"] += np.random.normal(0, 6.0, size=len(out_idx))

    df["residual"] = df["y_true"] - df["y_pred"]

    fig_line, fig_scatter = build_figures(df)

    out_dir.mkdir(parents=True, exist_ok=True)
    line_path = out_dir / "verified_line_plot.png"
    scatter_path = out_dir / "verified_pred_vs_true.png"

    # dpi~300 느낌: scale=3 + 충분한 픽셀 크기
    fig_line.write_image(line_path, width=1800, height=750, scale=3)
    fig_scatter.write_image(scatter_path, width=1000, height=1000, scale=3)

    return line_path, scatter_path
