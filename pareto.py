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


# === 기존 헬퍼 그대로 사용 ===
# _split_cols(df) -> (x_cols, f_cols, g_cols)
# _knee_indices_by_utopia(df, f_cols, k=3) -> np.ndarray
# _top_indices_by_sum(df, f_cols, top_n=5) -> np.ndarray

from pymoo.visualization.pcp import PCP
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Tuple

# ------------------ (1) 목적 PCP: 자동 강조(1순위=red, 2순위=blue) ------------------
def plot_objectives_pcp_pymoo(pareto_df: pd.DataFrame) -> None:
    """모든 f:: 축을 PCP로 표시.
    강조 대상은 내부에서 _knee / _top을 이용해 자동 선택:
      - 1순위: knee[0] (빨강)
      - 2순위: top[0]  (파랑, 1순위와 중복이면 다음 순번)
    """
    _, f_cols, _ = _split_cols(pareto_df)
    if not f_cols:
        raise ValueError("f:: 컬럼이 없습니다.")

    F = pareto_df[f_cols].to_numpy(float)

    # 우선순위 인덱스 선정: knee → top
    knees = _knee_indices_by_utopia(pareto_df, f_cols, k=3)
    tops  = _top_indices_by_sum(pareto_df, f_cols, top_n=3)

    order: List[int] = []
    if len(knees) > 0: order.append(int(knees[0]))
    if len(tops)  > 0:
        for t in tops:
            if int(t) not in order:
                order.append(int(t))
                break
    # 보호: 후보 부족 시 0번, 1번으로 보충
    if not order:
        order = [0]
    if len(order) == 1 and len(pareto_df) > 1:
        order.append(1)

    idx_red  = order[0]
    idx_blue = order[1] if len(order) > 1 else order[0]

    p = PCP(title="Objectives (PCP) | red=knee[0], blue=top[0]", labels=f_cols)
    p.add(F, color="lightgray", alpha=0.5, linewidth=1.0)      # 전체
    p.add(F[[idx_blue]], color="C0", alpha=0.95, linewidth=2.2) # 2순위=blue
    p.add(F[[idx_red]],  color="C3", alpha=1.0,  linewidth=2.8) # 1순위=red
    p.show()

    # (선택) 0~1 눈금 표시
    ax = plt.gca()
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{t:.2f}" for t in np.linspace(0, 1, 6)])
    plt.tight_layout()
    plt.show(block=True)


# ------------- (2) 선택 X PCP: 내부 자동 선택 + 동일 강조 규칙 ----------------
def plot_selected_x_pcp_pymoo(pareto_df: pd.DataFrame, k: int = 6) -> None:
    """분산 상위 k개의 x:: 축을 내부에서 자동 선택하여 PCP 표시.
    강조 대상은 Objectives PCP와 동일 규칙(1순위=red, 2순위=blue).
    """
    x_cols, f_cols, _ = _split_cols(pareto_df)
    if not x_cols:
        raise ValueError("x:: 컬럼이 없습니다.")
    if not f_cols:
        raise ValueError("f:: 컬럼이 없습니다.")

    # x 축 자동 선택(분산 상위 k)
    var = pareto_df[x_cols].var(numeric_only=True).sort_values(ascending=False)
    sel = var.index.tolist()[: max(1, min(k, len(var)))]

    X = pareto_df[sel].to_numpy(float)

    # 동일한 강조 인덱스 선정
    knees = _knee_indices_by_utopia(pareto_df, f_cols, k=3)
    tops  = _top_indices_by_sum(pareto_df, f_cols, top_n=3)

    order: List[int] = []
    if len(knees) > 0: order.append(int(knees[0]))
    if len(tops)  > 0:
        for t in tops:
            if int(t) not in order:
                order.append(int(t))
                break
    if not order:
        order = [0]
    if len(order) == 1 and len(pareto_df) > 1:
        order.append(1)

    idx_red  = order[0]
    idx_blue = order[1] if len(order) > 1 else order[0]

    p = PCP(title=f"Selected X (PCP) | red=knee[0], blue=top[0] | k={len(sel)}", labels=sel)
    p.add(X, color="lightgray", alpha=0.5, linewidth=1.0)       # 전체
    p.add(X[[idx_blue]], color="C0", alpha=0.95, linewidth=2.2) # 2순위=blue
    p.add(X[[idx_red]],  color="C3", alpha=1.0,  linewidth=2.8) # 1순위=red
    p.show()

    # (선택) 0~1 눈금 표시
    ax = plt.gca()
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{t:.2f}" for t in np.linspace(0, 1, 6)])
    plt.tight_layout()
    plt.show(block=True)

import numpy as np
import pandas as pd

def _apply_decision_to_base(x: np.ndarray, problem) -> pd.DataFrame:
    """중요 피처를 x로 고정하여 배치 입력을 구성."""
    df = problem.batch_base.copy()
    for j, f in enumerate(problem.important_features):
        df[f] = x[j]
    # 필요 시 스케일러 적용 (문제 클래스에 이미 동일 로직이 있다면 그것 사용)
    if getattr(problem, "scaler", None) is not None:
        arr = problem.scaler.transform(df[problem.feature_names])
        df = pd.DataFrame(arr, columns=problem.feature_names, index=problem.batch_base.index)
    return df

def _predict_generic(model, X: pd.DataFrame, feature_names) -> np.ndarray:
    """CatBoost/기타 모델 예측 래퍼(당신 코드에 이미 있다면 그것 사용)."""
    y = model.predict(X[feature_names])
    return np.asarray(y, dtype=float).reshape(-1)

def _compute_F_real_from_X(models: dict, problem, Xo: np.ndarray) -> np.ndarray:
    """선택된 해 Xo 각각에 대해, 실제 스케일의 MAE(타깃별 평균 |yhat-desired|) 행렬을 계산."""
    K = len(problem.targets)
    M = Xo.shape[0]
    F_real = np.zeros((M, K), dtype=float)

    for m in range(M):
        X_df = _apply_decision_to_base(Xo[m], problem)
        for k, t in enumerate(problem.targets):
            yhat = _predict_generic(models[t], X_df, problem.feature_names)
            desired = problem.specs[t]["desired"]    # 길이 N 벡터
            mae_real = np.abs(yhat - desired).mean() # ← 실제 단위의 평균 MAE
            F_real[m, k] = mae_real
    return F_real


def _collect_pareto(res, problem) -> pd.DataFrame:
    """
    res.opt (feasible nondominated set)에서 X를 꺼내,
    각 솔루션에 대해 '실제 단위' 평균 MAE를 재계산해 f::로 저장한다.
    - 타깃 순서는 problem.targets에 고정
    - 모델 접근은 problem.models[t] (러너/모델 혼용 대비)
    - 배치 입력 생성은 problem._apply_decision(x) (스케일러 포함)
    """
    opt = getattr(res, "opt", None)
    if opt is None:
        return pd.DataFrame()

    Xo, Fo, Go = opt.get("X"), opt.get("F"), opt.get("G")   # Fo(정규화 목적)는 표시용으로 쓰지 않음

    # --- 실제 단위 F 재계산 ---
    M = Xo.shape[0]
    K = len(problem.targets)
    F_real = np.zeros((M, K), dtype=float)

    for m in range(M):
        X_df = problem._apply_decision(Xo[m])  # 배치(all) 입력 생성 + (필요시) scaler 적용
        for k, t in enumerate(problem.targets):  # 순서 고정
            model_or_runner = problem.models[t]
            model = getattr(model_or_runner, "model", model_or_runner)  # 러너면 .model, 아니면 그대로
            yhat = _predict_generic(model, X_df, problem.feature_names)
            desired = problem.specs[t]["desired"]  # (N,) 벡터
            F_real[m, k] = float(np.mean(np.abs(yhat - desired)))  # 실제 단위 평균 MAE

    # --- DataFrame 구성 ---
    x_cols = [f"x::{f}" for f in problem.important_features]
    f_cols = [f"f::{name}" for name in getattr(problem, "objective_names", problem.targets)]

    dfs = [
        pd.DataFrame(Xo, columns=x_cols),
        pd.DataFrame(F_real, columns=f_cols),
    ]

    if Go is not None and Go.size > 0:
        g_cols = []
        for t in problem.targets:
            sp = problem.specs[t]
            if sp["lower"] is not None:
                g_cols.append(f"g::{t}_lower")
            if sp["upper"] is not None:
                g_cols.append(f"g::{t}_upper")
        dfs.append(pd.DataFrame(Go, columns=g_cols))

    df = pd.concat(dfs, axis=1).reset_index(drop=True)
    return df

