from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 내부 유틸: f::* 컬럼 자동 탐색 + F 행렬 준비
# ----------------------------------------------------------------------
def _get_f_cols(pareto_df: pd.DataFrame) -> List[str]:
    """pareto_df에서 f::* 형태의 목표 컬럼 목록을 찾는다.

    Args:
        pareto_df: _collect_pareto 결과 DataFrame.

    Returns:
        f_cols: 'f::' prefix를 가진 컬럼 이름 리스트.

    Raises:
        ValueError: f:: prefix를 가진 컬럼이 하나도 없을 때.
    """
    f_cols = [c for c in pareto_df.columns if c.startswith("f::")]
    if not f_cols:
        raise ValueError("pareto_df에 'f::'로 시작하는 목표 컬럼이 없습니다.")
    return f_cols


def _prepare_F(df: pd.DataFrame, f_cols: List[str]) -> np.ndarray:
    """목표 컬럼들로부터 numpy 행렬을 생성한다 (모두 minimize 기준).

    Args:
        df: 후보 해들을 포함한 DataFrame.
        f_cols: Pareto 비교에 사용할 목표 컬럼 이름 리스트.

    Returns:
        shape: (N, M) 의 numpy 배열.
    """
    return df[f_cols].to_numpy(dtype=float)


# ----------------------------------------------------------------------
# Pareto rank 계산 (non-dominated sorting)
# ----------------------------------------------------------------------
def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """a가 b를 Pareto 의미에서 지배하는지 여부를 반환한다 (모두 minimize 기준).

    Args:
        a: shape (M,) 의 목표값 벡터.
        b: shape (M,) 의 목표값 벡터.

    Returns:
        True이면 a가 b를 Pareto 정의상 지배(dominates)함.
    """
    # 모든 목표에서 a <= b 이고, 적어도 하나는 a < b
    return np.all(a <= b) and np.any(a < b)


def _pareto_ranks(df: pd.DataFrame, f_cols: List[str]) -> np.ndarray:
    """각 해의 Pareto rank(1,2,...)를 계산한다.

    Args:
        df: 후보 해들을 포함한 DataFrame (예: pareto_df).
        f_cols: Pareto 비교에 사용할 목표 컬럼 이름 리스트.

    Returns:
        길이 N의 numpy 배열. 각 원소는 해당 행의 Pareto rank (1부터 시작).
    """
    F = _prepare_F(df, f_cols)
    n = F.shape[0]

    if n == 0:
        return np.array([], dtype=int)

    # 각 점이 지배하는 점 목록, 지배당하는 개수
    S: List[List[int]] = [[] for _ in range(n)]
    n_dom = np.zeros(n, dtype=int)
    ranks = np.zeros(n, dtype=int)

    # 지배 관계 계산 (O(N^2))
    for p in range(n):
        for q in range(p + 1, n):
            if _dominates(F[p], F[q]):
                S[p].append(q)
                n_dom[q] += 1
            elif _dominates(F[q], F[p]):
                S[q].append(p)
                n_dom[p] += 1

    # 1차 front (지배당하지 않는 점들)
    current_front: List[int] = []
    for p in range(n):
        if n_dom[p] == 0:
            ranks[p] = 1
            current_front.append(p)

    # 나머지 front들 (껍질 벗기기)
    front_rank = 1
    while current_front:
        next_front: List[int] = []
        for p in current_front:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    ranks[q] = front_rank + 1
                    next_front.append(q)
        front_rank += 1
        current_front = next_front

    return ranks


# ----------------------------------------------------------------------
# 공개 함수 1: 대표 TOP N 선택 (score 컬럼 포함)
# ----------------------------------------------------------------------
def select_top_n(pareto_df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Pareto 1차 front에서 대표 N개 해를 선택한다.

    전략:
        1. pareto_df에서 f::* 목표 컬럼을 자동으로 찾는다.
        2. 해당 목표들 기준으로 Pareto rank(1,2,...)를 계산한다.
        3. 1차 front(ranks == 1)만 사용한다.
           - 만약 1차 front가 비어 있으면 전체를 사용한다.
        4. 각 목표를 [0,1]로 min-max 정규화한다 (0: best, 1: worst).
        5. 정규화된 값의 합을 score로 사용하고, score가 작은 순서대로 N개 선택한다.
        6. N보다 후보가 적으면 있는 만큼만 반환한다.

    주의:
        - 반환되는 DataFrame은 원본 pareto_df의 index를 그대로 유지한다.
        - 반환 DataFrame에는 score 컬럼이 포함된다.
        - pareto_df가 비어 있으면 빈 DataFrame을 그대로 반환한다.

    Args:
        pareto_df: _collect_pareto에서 생성된 Pareto 해 DataFrame.
        n: 선택할 대표 해 개수 (0 이하면 0으로 취급).

    Returns:
        대표 N개 해를 모은 DataFrame (원본 pareto_df의 부분집합).
    """
    if pareto_df.empty:
        return pareto_df.copy()

    # f::* 컬럼 자동 탐색
    f_cols = _get_f_cols(pareto_df)

    # Pareto rank 계산
    ranks = _pareto_ranks(pareto_df, f_cols)

    # 1차 front만 사용 (없으면 전체 사용)
    mask_front1 = ranks == 1
    if mask_front1.any():
        df_front = pareto_df.loc[mask_front1].copy()
    else:
        df_front = pareto_df.copy()

    total = len(df_front)
    if total == 0:
        return df_front

    # 사용할 개수 처리 (n이 total보다 크면 total까지)
    n_eff = max(int(n), 0)
    if n_eff == 0:
        # 동일한 컬럼 구조를 가진 빈 DataFrame 반환
        return df_front.iloc[0:0]

    k = min(n_eff, total)

    # 목표 행렬 준비
    F = _prepare_F(df_front, f_cols)

    # min-max 정규화 (0: best, 1: worst)
    f_min = F.min(axis=0)
    f_max = F.max(axis=0)
    denom = np.where(f_max > f_min, f_max - f_min, 1.0)  # 분모 0 방지
    F_norm = (F - f_min) / denom

    # score: 정규화된 값의 합 (작을수록 전체적으로 좋은 해)
    scores = F_norm.sum(axis=1)

    # score를 컬럼으로 추가 (index 정렬 그대로 유지)
    df_front = df_front.assign(score=scores)

    # score 오름차순으로 정렬 후 상위 k개 선택
    order = np.argsort(scores)
    selected_pos = order[:k]

    # iloc 사용해도 index는 원본 유지
    result = df_front.iloc[selected_pos]
    return result


# ----------------------------------------------------------------------
# 공개 함수 2: Knee point K개 선택 (knee_score 컬럼 포함)
# ----------------------------------------------------------------------
def _knee_points(pareto_df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Pareto 1차 front에서 knee point 후보 K개를 선택한다.

    간단한 heuristic:
        1. pareto_df에서 f::* 목표 컬럼을 자동으로 찾는다.
        2. Pareto rank를 계산해서 1차 front만 사용한다.
           - 만약 1차 front가 비어 있으면 전체를 사용한다.
        3. 각 목표를 [0,1]로 min-max 정규화한다.
           - 0: ideal (가장 좋은 값), 1: nadir (가장 나쁜 값).
        4. ideal = (0,...,0), nadir = (1,...,1)라고 보고,
           - d_ideal = ||F_norm - 0|| (ideal까지 거리)
           - d_nadir = ||F_norm - 1|| (nadir까지 거리)
           - knee_score = d_nadir / (d_ideal + eps)
        5. knee_score가 큰 점일수록
           "나쁜 쪽에서는 멀고, 좋은 쪽에는 상대적으로 가까운"
           무릎점에 가깝다고 보고 상위 K개 선택한다.
        6. K보다 후보가 적으면 있는 만큼만 반환한다.

    주의:
        - 반환되는 DataFrame은 원본 pareto_df의 index를 그대로 유지한다.
        - 반환 DataFrame에는 knee_score 컬럼이 포함된다.
        - pareto_df가 비어 있으면 빈 DataFrame을 그대로 반환한다.

    Args:
        pareto_df: _collect_pareto에서 생성된 Pareto 해 DataFrame.
        k: 선택할 knee point 개수 (0 이하면 0으로 취급).

    Returns:
        knee point 후보 K개를 모은 DataFrame (원본 pareto_df의 부분집합).
    """
    if pareto_df.empty:
        return pareto_df.copy()

    # f::* 컬럼 자동 탐색
    f_cols = _get_f_cols(pareto_df)

    # Pareto rank 계산
    ranks = _pareto_ranks(pareto_df, f_cols)

    # 1차 front만 사용 (없으면 전체 사용)
    mask_front1 = ranks == 1
    if mask_front1.any():
        df_front = pareto_df.loc[mask_front1].copy()
    else:
        df_front = pareto_df.copy()

    total = len(df_front)
    if total == 0:
        return df_front

    # 사용할 개수 처리
    k_eff = max(int(k), 0)
    if k_eff == 0:
        return df_front.iloc[0:0]

    kk = min(k_eff, total)

    # 목표 행렬 준비
    F = _prepare_F(df_front, f_cols)

    # [0,1] 정규화
    f_min = F.min(axis=0)
    f_max = F.max(axis=0)
    denom = np.where(f_max > f_min, f_max - f_min, 1.0)
    F_norm = (F - f_min) / denom

    # ideal = (0,...,0), nadir = (1,...,1)
    d_ideal = np.linalg.norm(F_norm, axis=1)
    d_nadir = np.linalg.norm(F_norm - 1.0, axis=1)

    eps = 1e-9
    knee_score = d_nadir / (d_ideal + eps)

    # knee_score를 컬럼으로 추가
    df_front = df_front.assign(knee_score=knee_score)

    # knee_score 내림차순으로 정렬 후 상위 kk개 선택
    order = np.argsort(-knee_score)
    selected_pos = order[:kk]

    result = df_front.iloc[selected_pos]
    return result


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

import numpy as np
import pandas as pd

def select_best_and_knee(pareto_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    pareto_df에서 Top-1(초록)과 Knee(빨강) 해를 1개씩 반환한다.
    f:: 컬럼은 반드시 존재해야 함.
    반환: (top_row, knee_row)
    """
    f_cols = [c for c in pareto_df.columns if c.startswith("f::")]
    if not f_cols:
        raise ValueError("pareto_df must contain f:: columns.")

    # --- ① Top 해 (전반적 오차 최소)
    rank_sum = pareto_df[f_cols].sum(axis=1)
    idx_top = rank_sum.idxmin()
    top_row = pareto_df.loc[idx_top]

    # --- ② Knee 해 (trade-off 절충점)
    F = pareto_df[f_cols].to_numpy(dtype=float)
    f_min, f_max = F.min(axis=0), F.max(axis=0)
    denom = np.where(f_max > f_min, f_max - f_min, 1.0)
    F_norm = (F - f_min) / denom
    dist = np.linalg.norm(F_norm, axis=1)
    idx_knee = np.argmin(dist)
    knee_row = pareto_df.iloc[idx_knee]

    return top_row, knee_row
