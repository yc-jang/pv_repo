from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# multi_objective_optimizer 모듈 재사용(타입/스펙)
from multi_objective_optimizer import TargetSpec, RunConfigMO


# =========================
# 내부 유틸
# =========================

def _predict_generic(model: Any, X: pd.DataFrame, feature_names: List[str]) -> np.ndarray:
    """CatBoost/XGBoost 예측 래퍼.

    Args:
        model: 학습된 모델 객체(.predict 지원 또는 xgboost.Booster)
        X: 입력 DataFrame(전체 feature_names 포함)
        feature_names: 모델이 학습에 사용한 컬럼 순서

    Returns:
        예측값 1D ndarray
    """
    try:
        y = model.predict(X[feature_names])
        return np.asarray(y, dtype=float).reshape(-1)
    except Exception:
        try:
            import xgboost as xgb
            dmat = xgb.DMatrix(X[feature_names], feature_names=feature_names)
            y = model.predict(dmat)
            return np.asarray(y, dtype=float).reshape(-1)
        except Exception as e:
            raise RuntimeError(f"Model prediction failed for type {type(model)}: {e}") from e


def _apply_decision_to_batch(
    base_df: pd.DataFrame,
    feature_names: List[str],
    important_features: List[str],
    x: np.ndarray,
) -> pd.DataFrame:
    """하나의 의사결정 벡터 x를 배치 전 행에 적용한 입력 프레임을 만든다."""
    df = base_df[feature_names].copy()
    # important 변수만 x로 대체
    for j, f in enumerate(important_features):
        df[f] = x[j]
    return df


def _bounds_from_train(
    X_train: pd.DataFrame,
    important_features: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """훈련분포 기반 바운드(min/max) 계산 (margin/scaler 없음)."""
    train_min = X_train[important_features].min(axis=0).astype(float)
    train_max = X_train[important_features].max(axis=0).astype(float)
    lb = train_min.to_numpy(dtype=float)
    ub = train_max.to_numpy(dtype=float)
    # 폭 0 보호
    eps = 1e-12
    tight = ub <= lb
    if np.any(tight):
        ub[tight] = lb[tight] + eps
    return lb, ub


def _random_in_bounds(
    lb: np.ndarray, ub: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """경계 내 균등 랜덤 샘플."""
    return lb + (ub - lb) * rng.random((n, lb.size))


# =========================
# Feasibility Probe
# =========================

def estimate_feasible_rate(
    models: Dict[str, Any],
    feature_names: List[str],
    important_features: List[str],
    X_train: pd.DataFrame,
    for_optimal: pd.DataFrame,
    target_specs: Dict[str, TargetSpec],
    samples: int,
    random_state: Optional[int] = 42,
) -> float:
    """임의 x 샘플에 대한 '배치 최악 제약' 기준의 feasible 비율을 추정한다.

    Args:
        models: 타깃명 -> 모델 객체
        feature_names: 전체 피처 이름(순서 포함)
        important_features: 의사결정 변수 피처 이름
        X_train: 훈련 DataFrame (바운드 산출용)
        for_optimal: 배치 입력(N행)
        target_specs: 타깃별 하한/상한/desired
        samples: 샘플 개수(M)
        random_state: 난수 시드

    Returns:
        0.0~1.0 사이의 추정 feasible 비율
    """
    rng = np.random.default_rng(random_state)
    lb, ub = _bounds_from_train(X_train, important_features)

    feas_count = 0
    N = for_optimal.shape[0]

    # 사전 전개: 타깃/스펙 배열화 (길이 N)
    specs_arr: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
    for t, sp in target_specs.items():
        def _to_arr(v: Optional[np.ndarray | pd.Series | float]) -> Optional[np.ndarray]:
            if v is None:
                return None
            a = v.to_numpy() if isinstance(v, (pd.Series, pd.Index)) else np.asarray(v)
            if a.ndim == 0:
                a = np.full(N, a.item(), dtype=float)
            if a.shape[0] != N:
                raise ValueError(f"[estimate_feasible_rate] Spec length mismatch for {t}: N={N}, got={a.shape[0]}")
            return a.astype(float)

        specs_arr[t] = {
            "lower": _to_arr(sp.lower),
            "upper": _to_arr(sp.upper),
            "desired": _to_arr(sp.desired),  # 목적에는 쓰지 않지만 형상 확인 겸 캐시
        }

    for _ in range(samples):
        x = _random_in_bounds(lb, ub, 1, rng)[0]
        X = _apply_decision_to_batch(for_optimal, feature_names, important_features, x)

        feasible = True
        for t, model in models.items():
            yhat = _predict_generic(model, X, feature_names)
            sp = specs_arr[t]
            if sp["lower"] is not None:
                # 최악 위반(max): lower - yhat <= 0 이어야 함
                if np.max(sp["lower"] - yhat) > 0:
                    feasible = False
                    break
            if sp["upper"] is not None:
                # 최악 위반(max): yhat - upper <= 0 이어야 함
                if np.max(yhat - sp["upper"]) > 0:
                    feasible = False
                    break
        feas_count += int(feasible)

    r = feas_count / float(samples)
    logger.info(f"[probe] feasible_rate r={r:.4f} with samples={samples}")
    return r


# =========================
# 오토튜너
# =========================

@dataclass(frozen=True)
class AutoTuneCaps:
    """오토튜너 상/하한 및 배수 설정."""
    pop_cap_max: int = 600       # pop 상한
    pop_cap_min: int = 80        # pop 하한
    gen_cap_max: int = 300       # gen 상한
    gen_cap_min: int = 20        # gen 하한


def autotune_runconfig_mo(
    for_optimal: pd.DataFrame,
    important_features: List[str],
    targets: List[str],
    feasible_rate: Optional[float] = None,
    *,
    base_pop_formula: Optional[callable] = None,
    base_eval_budget_formula: Optional[callable] = None,
    caps: AutoTuneCaps = AutoTuneCaps(),
) -> RunConfigMO:
    """NSGA-II 파라미터(RunConfigMO)를 데이터 규모와 난이도에 맞춰 산출한다.

    Args:
        for_optimal: 배치 입력(N행)
        important_features: 의사결정 변수 리스트
        targets: 타깃명 리스트(목적 수 K)
        feasible_rate: 프로브로 추정한 feasible 비율 r (None이면 보수적 기본값으로 간주)
        base_pop_formula: pop_size 기본 공식 커스터마이저 (d, K) -> pop_size
        base_eval_budget_formula: 총 평가예산 공식 커스터마이저 (d, K) -> E
        caps: 상/하한 캡

    Returns:
        RunConfigMO(pop_size, n_gen, top_n, random_seed, log_every)
    """
    N = int(for_optimal.shape[0])
    d = int(len(important_features))
    K = int(len(targets))
    logger.info(f"[autotune] N={N}, d={d}, K={K}, r={feasible_rate}")

    # --- 기본 공식 ---
    if base_pop_formula is None:
        def base_pop_formula(d_: int, K_: int) -> int:
            # max(100, round(12*d*(1 + 0.15*K))) capped later
            return max(100, int(round(12 * d_ * (1.0 + 0.15 * K_))))
    if base_eval_budget_formula is None:
        def base_eval_budget_formula(d_: int, K_: int) -> int:
            # E = 4000 + 250*d + 300*K
            return int(4000 + 250 * d_ + 300 * K_)

    pop = base_pop_formula(d, K)
    E = base_eval_budget_formula(d, K)

    # --- r 기반 스케일링 ---
    if feasible_rate is None:
        scale_pop, scale_gen = 1.0, 1.0  # 보수적 기본
    else:
        r = feasible_rate
        if r < 0.01:
            scale_pop, scale_gen = 2.0, 1.5   # 총 ~3배
        elif r < 0.10:
            scale_pop, scale_gen = 1.5, 1.25  # 총 ~1.9배
        elif r >= 0.30:
            scale_pop, scale_gen = 0.8, 0.8   # 총 ~0.64배
        else:
            scale_pop, scale_gen = 1.0, 1.0

    pop = int(pop * scale_pop)
    # 상/하한 캡
    pop = int(np.clip(pop, caps.pop_cap_min, caps.pop_cap_max))

    # n_gen 계산 (먼저 기본 gen, 이후 r 스케일 적용)
    gen = ceil(E / pop)
    gen = int(np.clip(int(gen * scale_gen), caps.gen_cap_min, caps.gen_cap_max))

    # top_n은 pop의 1/3, 최대 50
    top_n = int(min(50, max(5, pop // 3)))

    cfg = RunConfigMO(
        pop_size=pop,
        n_gen=gen,
        top_n=top_n,
        random_seed=42,
        log_every=10,
    )
    logger.info(f"[autotune] -> pop={cfg.pop_size}, gen={cfg.n_gen}, top_n={cfg.top_n}")
    return cfg


def autotune_with_probe(
    models: Dict[str, Any],
    feature_names: List[str],
    important_features: List[str],
    X_train: pd.DataFrame,
    for_optimal: pd.DataFrame,
    target_specs: Dict[str, TargetSpec],
    targets: List[str],
    *,
    sample_factor: float = 30.0,
    min_samples: int = 200,
    random_state: Optional[int] = 42,
    caps: AutoTuneCaps = AutoTuneCaps(),
) -> RunConfigMO:
    """프로브 r을 먼저 추정한 뒤, r을 반영해 RunConfigMO를 산출한다.

    샘플 개수 M = max(min_samples, round(sample_factor * d))

    Args:
        models: 타깃명 -> 모델 객체
        feature_names: 전체 피처 이름(순서 포함)
        important_features: 의사결정 변수 리스트
        X_train: 훈련 DataFrame (바운드 산출용)
        for_optimal: 배치 입력(N행)
        target_specs: 타깃별 스펙
        targets: 타깃명 리스트
        sample_factor: d에 곱해 프로브 샘플 수를 정하는 계수(기본 30)
        min_samples: 프로브 최소 샘플 수(기본 200)
        random_state: 난수 시드
        caps: 캡 설정

    Returns:
        RunConfigMO
    """
    d = len(important_features)
    M = max(int(min_samples), int(round(sample_factor * d)))
    r = estimate_feasible_rate(
        models=models,
        feature_names=feature_names,
        important_features=important_features,
        X_train=X_train,
        for_optimal=for_optimal,
        target_specs=target_specs,
        samples=M,
        random_state=random_state,
    )
    return autotune_runconfig_mo(
        for_optimal=for_optimal,
        important_features=important_features,
        targets=targets,
        feasible_rate=r,
        caps=caps,
    )
