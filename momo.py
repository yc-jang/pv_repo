from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Mapping, Sequence, Union
import os
import pickle
import pandas as pd
import numpy as np

# ---- 외부 구현 클래스(실제 모듈/클래스가 존재한다고 가정) ----
from bayesian_optimizer import BayesianOptimzation
from genetic_optimizer import GeneticOptimizer
from catboost_runner import CatBoostRunner
from tabpfn_runner import TabPFNRunner

# ---- 다목적 최적화용 (pymoo) ----
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination


# ============================== 다목적 스펙 구조 ==============================

@dataclass(frozen=True)
class TargetSpec:
    """다목적 타깃별 규격/목표 정의.

    Attributes:
        lower: 하한 (스칼라 or 길이 N 벡터), 없으면 None
        upper: 상한 (스칼라 or 길이 N 벡터), 없으면 None
        desired: 목표값 (스칼라 or 길이 N 벡터), 필수
        spec_range_fallback: 상/하한 중 하나라도 없을 때 정규화 분모로 쓸 값(예: IQR). None이면 1.0
    """
    lower: Optional[Union[float, np.ndarray, pd.Series]] = None
    upper: Optional[Union[float, np.ndarray, pd.Series]] = None
    desired: Optional[Union[float, np.ndarray, pd.Series]] = None
    spec_range_fallback: Optional[float] = None


# ============================== 설정/유틸 ==============================

@dataclass(frozen=True)
class BuildConfig:
    """파이프라인 빌드(준비) 단계 설정.

    Args:
        base_dir: 프로젝트 루트 디렉터리(예: Path(__file__).parent).
        data_pickle: end2end 데이터 피클 경로.
        target: (단일목적) CatBoost 모델 선택에 사용할 TARGET 문자열(파일명 부분 일치).
        column_sets: {'only_cond':[...], 'matr_cond':[...]} 형태의 컬럼셋.
        column_set_key: important_feature를 고를 키(예: 'only_cond' 또는 'matr_cond').
        desired_value: (단일목적) 목표 스칼라 값.
        feature_names: (선택) 학습에 사용한 전체 피처 리스트. 미지정 시 모델/데이터로부터 추정.
        # ---- 다목적 전용 선택 인자 ----
        targets: Optional[List[str]] = None
            # 다목적 타깃명 리스트(각 타깃명이 CatBoost .pkl 파일명에 포함된다고 가정).
        multi_target_specs: Optional[Dict[str, TargetSpec]] = None
            # 타깃명->TargetSpec 매핑. 다목적 실행 시 필수.
    """
    base_dir: Path
    data_pickle: Path
    target: str
    column_sets: Dict[str, List[str]]
    column_set_key: str
    desired_value: float
    feature_names: Optional[List[str]] = None
    targets: Optional[List[str]] = None
    multi_target_specs: Optional[Dict[str, TargetSpec]] = None


@dataclass(frozen=True)
class RunConfigBO:
    """베이지안 최적화 실행 파라미터(호출 시점에 전달)."""
    init_points: int
    n_iter: int


@dataclass(frozen=True)
class RunConfigGA:
    """유전 알고리즘 실행 파라미터(호출 시점에 전달)."""
    population: int
    parents: int
    generations: int


@dataclass(frozen=True)
class RunConfigMO:
    """다목적(NSGA-II) 실행 파라미터."""
    pop_size: int = 120
    n_gen: int = 150
    top_n: int = 20
    random_seed: Optional[int] = 42
    log_every: int = 10


@dataclass(frozen=True)
class BuildArtifacts:
    """빌드 단계에서 준비된 산출물.

    Attributes:
        df: 로드된 전체 DataFrame.
        # --- 단일목적 호환 ---
        cat_model_obj: 최신 CatBoost 단일모델 (없으면 None).
        tabpfn_model_obj: 최신 TabPFN 단일모델 (없으면 None).
        # --- 다목적 확장 ---
        cat_models: 타깃명->CatBoost 모델 객체 매핑(없으면 None).
        tabpfn_models: 타깃명->TabPFN  모델 객체 매핑(없으면 None).
        feature_names: 학습에 사용한 전체 피처 리스트.
        important_features: 선택된 column_set_key에 해당하는 중요 컬럼 리스트.
        catboost_model_path: 단일 CatBoost .pkl 상대경로(없으면 None).
        tabpfn_ckpt_path: 단일 TabPFN .ckpt 상대경로(없으면 None).
        catboost_model_paths: 타깃명->상대경로(없으면 None).
        tabpfn_ckpt_paths: 타깃명->상대경로(없으면 None).
    """
    df: pd.DataFrame
    cat_model_obj: Optional[Any]
    tabpfn_model_obj: Optional[Any]
    cat_models: Optional[Dict[str, Any]]
    tabpfn_models: Optional[Dict[str, Any]]
    feature_names: List[str]
    important_features: List[str]
    catboost_model_path: Optional[str]
    tabpfn_ckpt_path: Optional[str]
    catboost_model_paths: Optional[Dict[str, str]]
    tabpfn_ckpt_paths: Optional[Dict[str, str]]


def to_relpath(path: Path, start: Optional[Path] = None) -> str:
    """Path를 OS 상대 경로 문자열로 변환. 드라이브가 다르거나 기준 불가 시 절대경로."""
    base = Path.cwd() if start is None else Path(start)
    try:
        return os.path.relpath(Path(path).resolve(), start=base.resolve())
    except ValueError:
        return str(Path(path).resolve())


def load_pickle_df(pickle_path: Path) -> pd.DataFrame:
    """피클에서 pandas DataFrame을 로드."""
    if not pickle_path.exists():
        raise FileNotFoundError(f"pickle not found: {pickle_path}")
    with open(pickle_path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, pd.DataFrame):
        raise ValueError(f"Loaded object is not a DataFrame: type={type(obj)}")
    return obj


def find_model_paths(models_dir: Path, target: str) -> Tuple[Optional[Path], Optional[Path]]:
    """models 아래에서 CatBoost(.pkl), TabPFN(.ckpt) 경로를 최신 mtime 기준으로 선택."""
    if not models_dir.exists():
        return None, None

    target_lower = target.lower()
    cat_candidates: List[Path] = []
    tabpfn_candidates: List[Path] = []

    for p in models_dir.rglob("*"):
        if not p.is_file():
            continue
        name_lower = p.name.lower()
        if p.suffix == ".pkl" and target_lower in name_lower:
            cat_candidates.append(p)
        if p.suffix == ".ckpt" and "tabpfn" in name_lower:
            tabpfn_candidates.append(p)

    def _pick_latest(paths: List[Path]) -> Optional[Path]:
        return max(paths, key=lambda x: x.stat().st_mtime) if paths else None

    return _pick_latest(cat_candidates), _pick_latest(tabpfn_candidates)


def find_models_by_targets(models_dir: Path, targets: List[str]) -> Tuple[Dict[str, Optional[Path]], Dict[str, Optional[Path]]]:
    """타깃명 리스트에 대해 각 타깃명을 파일명에 포함하는 최신 모델 경로 매핑을 찾는다."""
    cat_map: Dict[str, Optional[Path]] = {}
    tab_map: Dict[str, Optional[Path]] = {}

    for tgt in targets:
        cat_map[tgt], tab_map[tgt] = find_model_paths(models_dir, target=tgt)
    return cat_map, tab_map


def _infer_feature_names(
    preferred: Optional[List[str]],
    cat_runner: Optional[CatBoostRunner],
    df: Optional[pd.DataFrame],
) -> List[str]:
    """feature_names 추정: 우선순위 = preferred → cat_runner 내부 정보 → df.columns."""
    if preferred is not None:
        return list(preferred)
    if cat_runner is not None:
        model_obj = getattr(cat_runner, "model", None)
        if model_obj is not None:
            for attr in ("feature_names_", "feature_names"):
                names = getattr(model_obj, attr, None)
                if names is not None:
                    return list(names)
        for attr in ("X_train", "X_test"):
            x_obj = getattr(cat_runner, attr, None)
            if isinstance(x_obj, pd.DataFrame):
                return x_obj.columns.tolist()
    if df is not None:
        return df.columns.tolist()
    raise ValueError("Unable to infer feature_names; please provide BuildConfig.feature_names.")


def _extract_model_object(runner_or_none: Optional[Any]) -> Optional[Any]:
    """러너 인스턴스에서 실제 모델 객체를 추출(.model), 없으면 러너 자체를 반환."""
    if runner_or_none is None:
        return None
    return getattr(runner_or_none, "model", runner_or_none)


# ============================== 빌드 단계 ==============================

def build_pipeline(cfg: BuildConfig) -> BuildArtifacts:
    """데이터/모델/피처/중요컬럼을 준비하여 런 전 상태로 만든다."""
    base_dir = cfg.base_dir.resolve()
    models_dir = (base_dir / "models").resolve()
    data_pickle = cfg.data_pickle.resolve()

    # 1) 데이터 로드
    df = load_pickle_df(data_pickle)

    # 2) (단일) 모델 경로 탐색 후 로드
    cat_path_single, tab_path_single = find_model_paths(models_dir=models_dir, target=cfg.target)

    cat_runner_single: Optional[CatBoostRunner] = None
    tab_runner_single: Optional[TabPFNRunner] = None

    if cat_path_single is not None:
        cat_runner_single = CatBoostRunner().load_model(cat_path_single)
    if tab_path_single is not None:
        tab_runner_single = TabPFNRunner().load_model(tab_path_single)

    # 3) feature_names 결정
    feature_names = _infer_feature_names(cfg.feature_names, cat_runner_single, df)

    # 4) important_features 결정
    if cfg.column_set_key not in cfg.column_sets:
        raise KeyError(f"column_sets does not contain key='{cfg.column_set_key}'")
    important_features = cfg.column_sets[cfg.column_set_key]
    if not isinstance(important_features, list):
        raise ValueError(f"column_sets['{cfg.column_set_key}'] must be a list of column names")

    # 5) (다목적) 타깃별 모델 경로 탐색 및 로드 (옵션)
    cat_models: Optional[Dict[str, Any]] = None
    tab_models: Optional[Dict[str, Any]] = None
    cat_paths_map: Optional[Dict[str, str]] = None
    tab_paths_map: Optional[Dict[str, str]] = None

    if cfg.targets:
        cat_paths, tab_paths = find_models_by_targets(models_dir, cfg.targets)
        cat_models = {}
        tab_models = {}
        cat_paths_map = {}
        tab_paths_map = {}

        for tgt in cfg.targets:
            cp = cat_paths.get(tgt)
            tp = tab_paths.get(tgt)
            if cp is not None:
                cat_models[tgt] = _extract_model_object(CatBoostRunner().load_model(cp))
                cat_paths_map[tgt] = to_relpath(cp)
            else:
                cat_models[tgt] = None
            if tp is not None:
                tab_models[tgt] = _extract_model_object(TabPFNRunner().load_model(tp))
                tab_paths_map[tgt] = to_relpath(tp)
            else:
                tab_models[tgt] = None

        # None 모델만 있는 경우 clean-up
        if all(v is None for v in cat_models.values()):
            cat_models = None
        if all(v is None for v in tab_models.values()):
            tab_models = None
        if not cat_paths_map:
            cat_paths_map = None
        if not tab_paths_map:
            tab_paths_map = None

    # 6) 모델 오브젝트 추출(.model 있으면 추출) - 단일
    cat_model_obj = _extract_model_object(cat_runner_single)
    tabpfn_model_obj = _extract_model_object(tab_runner_single)

    return BuildArtifacts(
        df=df,
        cat_model_obj=cat_model_obj,
        tabpfn_model_obj=tabpfn_model_obj,
        cat_models=cat_models,
        tabpfn_models=tab_models,
        feature_names=feature_names,
        important_features=important_features,
        catboost_model_path=to_relpath(cat_path_single) if cat_path_single else None,
        tabpfn_ckpt_path=to_relpath(tab_path_single) if tab_path_single else None,
        catboost_model_paths=cat_paths_map,
        tabpfn_ckpt_paths=tab_paths_map,
    )


# ============================== (단일목적) 런 단계 ==============================

@dataclass
class OptimizerPack:
    """동일 인터페이스로 초기화된 최적화기 패키지.

    Attributes:
        bo_cat: CatBoost 모델을 쓴 BO (없을 수 있음)
        bo_tab: TabPFN  모델을 쓴 BO (없을 수 있음)
        ga_cat: CatBoost 모델을 쓴 GA (없을 수 있음)
        ga_tab: TabPFN  모델을 쓴 GA (없을 수 있음)
    """
    bo_cat: Optional[BayesianOptimzation] = None
    bo_tab: Optional[BayesianOptimzation] = None
    ga_cat: Optional[GeneticOptimizer] = None
    ga_tab: Optional[GeneticOptimizer] = None


def make_optimizers(
    artifacts: BuildArtifacts,
    build_cfg: BuildConfig,
    use_catboost: bool = True,
    use_tabpfn: bool = False,
    make_bayes: bool = True,
    make_genetic: bool = True,
) -> OptimizerPack:
    """빌드 산출물로부터 (단일목적) 옵티마이저 인스턴스들을 구성한다."""
    pack = OptimizerPack()

    if use_catboost and artifacts.cat_model_obj is not None:
        if make_bayes:
            pack.bo_cat = BayesianOptimzation(
                model=artifacts.cat_model_obj,
                model_type="catboost",
                feature_names=artifacts.feature_names,
                desired_value=build_cfg.desired_value,
                important_feature=artifacts.important_features,
            )
        if make_genetic:
            pack.ga_cat = GeneticOptimizer(
                model=artifacts.cat_model_obj,
                model_type="catboost",
                feature_names=artifacts.feature_names,
                desired_value=build_cfg.desired_value,
                important_feature=artifacts.important_features,
            )

    if use_tabpfn and artifacts.tabpfn_model_obj is not None:
        if make_bayes:
            pack.bo_tab = BayesianOptimzation(
                model=artifacts.tabpfn_model_obj,
                model_type="tabpfn",
                feature_names=artifacts.feature_names,
                desired_value=build_cfg.desired_value,
                important_feature=artifacts.important_features,
            )
        if make_genetic:
            pack.ga_tab = GeneticOptimizer(
                model=artifacts.tabpfn_model_obj,
                model_type="tabpfn",
                feature_names=artifacts.feature_names,
                desired_value=build_cfg.desired_value,
                important_feature=artifacts.important_features,
            )

    return pack


def run_bayes(optimizer: BayesianOptimzation, run_cfg: RunConfigBO) -> Tuple[Any, float]:
    """베이지안 최적화 실행."""
    return optimizer.optimize(init_points=run_cfg.init_points, n_iter=run_cfg.n_iter)


def run_genetic(optimizer: GeneticOptimizer, run_cfg: RunConfigGA) -> Tuple[Any, float]:
    """유전 알고리즘 실행."""
    return optimizer.optimize(
        population=run_cfg.population,
        parents=run_cfg.parents,
        generations=run_cfg.generations,
    )


def build_then_run(
    build_cfg: BuildConfig,
    bo_run_cfg: Optional[RunConfigBO] = None,
    ga_run_cfg: Optional[RunConfigGA] = None,
    use_catboost: bool = True,
    use_tabpfn: bool = False,
    do_bayes: bool = True,
    do_genetic: bool = True,
) -> Dict[str, Any]:
    """한 번에 빌드하고 원하는 조합으로 (단일목적) 런까지 수행."""
    artifacts = build_pipeline(build_cfg)
    pack = make_optimizers(
        artifacts=artifacts,
        build_cfg=build_cfg,
        use_catboost=use_catpfn:=use_catboost,
        use_tabpfn=use_tabpfn,
        make_bayes=do_bayes,
        make_genetic=do_genetic,
    )

    results: Dict[str, Optional[Tuple[Any, float]]] = {
        "bo_cat": None,
        "bo_tab": None,
        "ga_cat": None,
        "ga_tab": None,
    }

    if do_bayes and bo_run_cfg is not None:
        if pack.bo_cat is not None:
            results["bo_cat"] = run_bayes(pack.bo_cat, bo_run_cfg)
        if pack.bo_tab is not None:
            results["bo_tab"] = run_bayes(pack.bo_tab, bo_run_cfg)

    if do_genetic and ga_run_cfg is not None:
        if pack.ga_cat is not None:
            results["ga_cat"] = run_genetic(pack.ga_cat, ga_run_cfg)
        if pack.ga_tab is not None:
            results["ga_tab"] = run_genetic(pack.ga_tab, ga_run_cfg)

    return {
        "artifacts": artifacts,
        "results": results,
    }


# ============================== 다목적 최적화 (pymoo) ==============================

def _to_array(v: Optional[Union[float, np.ndarray, pd.Series]], n: int) -> Optional[np.ndarray]:
    """스칼라/시리즈/ndarray를 길이 n의 ndarray로 정규화."""
    if v is None:
        return None
    if isinstance(v, (pd.Series, pd.Index)):
        a = v.to_numpy()
    else:
        a = np.asarray(v)
    if a.ndim == 0:
        a = np.full(n, a.item(), dtype=float)
    if a.shape[0] != n:
        raise ValueError(f"Spec vector length mismatch: expected {n}, got {a.shape[0]}")
    return a.astype(float)


def _compute_bounds_with_margin(train_min: float, train_max: float, safety_margin: float = 0.05) -> Tuple[float, float]:
    """훈련 범위를 기준으로 ±margin 완화한 바운드 산출."""
    lb = float(train_min * (1.0 - safety_margin))
    ub = float(train_max * (1.0 + safety_margin))
    if not np.isfinite(lb) or not np.isfinite(ub):
        raise ValueError("Non-finite bounds encountered.")
    if ub <= lb:
        eps = 1e-6 if lb == 0 else abs(lb) * 1e-6
        ub = lb + eps
    return lb, ub


def _predict_generic(model: Any, X: pd.DataFrame, feature_names: List[str]) -> np.ndarray:
    """CatBoost/XGBoost 모델 예측 래퍼."""
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


class BatchMultiObjectiveProblem(Problem):
    """배치(all) 기반 다목적 최적화 문제.

    하나의 조절 벡터 x(important features)로 for_optimal의 모든 행(N개)에 동일 적용하여,
    타깃별 정규화 MAE(배치 평균)를 목적(f1..fK)으로, 하한/상한 위반(배치 최악)을 제약으로 구성.
    """

    def __init__(
        self,
        models: Dict[str, Any],
        feature_names: List[str],
        important_features: List[str],
        X_train: pd.DataFrame,
        for_optimal: pd.DataFrame,
        target_specs: Dict[str, TargetSpec],
        scaler: Optional[Any] = None,
        safety_margin: float = 0.05,
        objective_names: Optional[List[str]] = None,
    ) -> None:
        self.models = models
        self.targets = list(models.keys())
        self.feature_names = feature_names
        self.important_features = important_features
        self.scaler = scaler

        # 검증
        if not set(self.important_features).issubset(self.feature_names):
            raise ValueError("important_features must be a subset of feature_names.")
        if not set(self.feature_names).issubset(for_optimal.columns):
            raise ValueError("for_optimal must include all feature_names columns.")
        if not set(self.feature_names).issubset(X_train.columns):
            raise ValueError("X_train must include all feature_names columns.")
        if set(self.targets) != set(target_specs.keys()):
            raise ValueError("target_specs must be provided for all targets exactly.")

        self.N = int(for_optimal.shape[0])
        self.K = len(self.targets)
        self.objective_names = objective_names if objective_names is not None else self.targets

        # 바운드
        train_min = X_train[self.important_features].min(axis=0).astype(float)
        train_max = X_train[self.important_features].max(axis=0).astype(float)
        lbs, ubs = [], []
        for f in self.important_features:
            lb, ub = _compute_bounds_with_margin(train_min[f], train_max[f], safety_margin)
            lbs.append(lb)
            ubs.append(ub)
        self.xl = np.array(lbs, dtype=float)
        self.xu = np.array(ubs, dtype=float)

        # 배치 입력 고정
        self.batch_base = for_optimal[self.feature_names].copy()

        # 스펙/정규화 분모 구성
        self.specs: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
        for t in self.targets:
            sp = target_specs[t]
            lower = _to_array(sp.lower, self.N)
            upper = _to_array(sp.upper, self.N)
            desired = _to_array(sp.desired, self.N)
            if desired is None:
                raise ValueError(f"Target '{t}' requires desired values.")

            if lower is not None and upper is not None:
                denom = (upper - lower).astype(float)
                denom[~np.isfinite(denom) | (denom <= 0)] = np.nan
                if sp.spec_range_fallback is not None:
                    denom = np.where(np.isfinite(denom), denom, float(sp.spec_range_fallback))
                else:
                    denom = np.where(np.isfinite(denom), denom, 1.0)
            else:
                denom = np.full(self.N, float(sp.spec_range_fallback) if sp.spec_range_fallback else 1.0, dtype=float)
            denom[denom == 0] = 1.0

            self.specs[t] = {"lower": lower, "upper": upper, "desired": desired, "denom": denom}

        super().__init__(
            n_var=len(self.important_features),
            n_obj=self.K,
            n_constr=self._count_constraints(),
            xl=self.xl,
            xu=self.xu,
            elementwise_evaluation=True,
        )

    def _count_constraints(self) -> int:
        c = 0
        for t in self.targets:
            sp = self.specs[t]
            if sp["lower"] is not None:
                c += 1
            if sp["upper"] is not None:
                c += 1
        return c

    def _apply_decision(self, x: np.ndarray) -> pd.DataFrame:
        df = self.batch_base.copy()
        for j, f in enumerate(self.important_features):
            df[f] = x[j]
        if self.scaler is not None:
            arr = self.scaler.transform(df[self.feature_names])
            df = pd.DataFrame(arr, columns=self.feature_names, index=self.batch_base.index)
        return df

    def _evaluate_elementwise(self, x: np.ndarray, out: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        X = self._apply_decision(x)

        preds: Dict[str, np.ndarray] = {}
        for t, model in self.models.items():
            if model is None:
                raise RuntimeError(f"Model for target '{t}' is None.")
            preds[t] = _predict_generic(model, X, self.feature_names)

        F = np.zeros(self.K, dtype=float)
        G_vals: List[float] = []

        for k, t in enumerate(self.targets):
            sp = self.specs[t]
            yhat = preds[t]
            desired = sp["desired"]
            denom = sp["denom"]

            mae = np.abs(yhat - desired) / denom
            F[k] = float(np.nanmean(mae))

            if sp["lower"] is not None:
                G_vals.append(float(np.max(sp["lower"] - yhat)))
            if sp["upper"] is not None:
                G_vals.append(float(np.max(yhat - sp["upper"])))

        out["F"] = F
        if self.n_constr > 0:
            out["G"] = np.array(G_vals, dtype=float)


class ProgressCallback(Callback):
    """세대별 진행 상황 간단 로깅."""
    def __init__(self, log_every: int = 10) -> None:
        super().__init__()
        self.log_every = log_every

    def notify(self, algorithm) -> None:
        gen = algorithm.n_gen
        if gen % self.log_every == 0 or gen == 1:
            print(f"[NSGA-II] gen={gen}, pop={algorithm.pop_size}")


@dataclass
class MOResult:
    """다목적 결과 컨테이너(경량)."""
    pareto_df: pd.DataFrame
    top_df: pd.DataFrame
    knee_df: pd.DataFrame
    x_cols: List[str]
    f_cols: List[str]
    g_cols: List[str]


class MultiObjectiveOptimizer:
    """NSGA-II 기반 다목적 옵티마이저(ALL 모드)."""

    def __init__(
        self,
        models: Dict[str, Any],
        feature_names: List[str],
        important_features: List[str],
        X_train: pd.DataFrame,
        scaler: Optional[Any] = None,
        safety_margin: float = 0.05,
    ) -> None:
        self.models = models
        self.feature_names = feature_names
        self.important_features = important_features
        self.X_train = X_train[self.feature_names].copy()
        self.scaler = scaler
        self.safety_margin = safety_margin

    def _build_problem(
        self,
        for_optimal: pd.DataFrame,
        target_specs: Dict[str, TargetSpec],
        objective_names: Optional[List[str]] = None,
    ) -> BatchMultiObjectiveProblem:
        return BatchMultiObjectiveProblem(
            models=self.models,
            feature_names=self.feature_names,
            important_features=self.important_features,
            X_train=self.X_train,
            for_optimal=for_optimal[self.feature_names].copy(),
            target_specs=target_specs,
            scaler=self.scaler,
            safety_margin=self.safety_margin,
            objective_names=objective_names,
        )

    @staticmethod
    def _nsga2(pop_size: int = 120) -> NSGA2:
        return NSGA2(pop_size=pop_size)

    @staticmethod
    def _termination(n_gen: int = 150):
        return get_termination("n_gen", n_gen)

    @staticmethod
    def _collect_pareto(res, problem: BatchMultiObjectiveProblem) -> pd.DataFrame:
        X = res.X
        F = res.F
        G = res.G if hasattr(res, "G") and res.G is not None else None

        x_cols = [f"x::{f}" for f in problem.important_features]
        f_cols = [f"f::{name}" for name in problem.objective_names]
        df_x = pd.DataFrame(X, columns=x_cols)
        df_f = pd.DataFrame(F, columns=f_cols)

        dfs = [df_x, df_f]
        g_cols: List[str] = []
        if G is not None and G.size > 0:
            g_cols = [f"g::{i+1}" for i in range(G.shape[1])]
            df_g = pd.DataFrame(G, columns=g_cols)
            dfs.append(df_g)

        return pd.concat(dfs, axis=1)

    @staticmethod
    def _select_top_n(pareto_df: pd.DataFrame, f_cols: List[str], n: int = 20) -> pd.DataFrame:
        ranks = pareto_df[f_cols].sum(axis=1)
        return pareto_df.assign(_rank=ranks).sort_values("_rank").head(n).drop(columns=["_rank"])

    @staticmethod
    def _knee_points(pareto_df: pd.DataFrame, f_cols: List[str], k: int = 3) -> pd.DataFrame:
        F = pareto_df[f_cols].to_numpy(dtype=float)
        f_min = F.min(axis=0)
        f_max = F.max(axis=0)
        denom = np.where(f_max > f_min, (f_max - f_min), 1.0)
        F_norm = (F - f_min) / denom
        dist = np.linalg.norm(F_norm, axis=1)
        idx = np.argsort(dist)[: max(1, min(k, len(dist)))]
        return pareto_df.iloc[idx].copy()

    def optimize(
        self,
        for_optimal: pd.DataFrame,
        target_specs: Dict[str, TargetSpec],
        objective_names: Optional[List[str]] = None,
        pop_size: int = 120,
        n_gen: int = 150,
        random_seed: Optional[int] = 42,
        log_every: int = 10,
        top_n: int = 20,
    ) -> MOResult:
        problem = self._build_problem(for_optimal, target_specs, objective_names)
        alg = self._nsga2(pop_size=pop_size)
        term = self._termination(n_gen=n_gen)
        cb = ProgressCallback(log_every=log_every)

        res = minimize(
            problem,
            alg,
            term,
            seed=random_seed,
            callback=cb,
            save_history=False,
            verbose=False,
        )

        pareto_df = self._collect_pareto(res, problem)
        x_cols = [c for c in pareto_df.columns if c.startswith("x::")]
        f_cols = [c for c in pareto_df.columns if c.startswith("f::")]
        g_cols = [c for c in pareto_df.columns if c.startswith("g::")]

        top_df = self._select_top_n(pareto_df, f_cols, n=top_n)
        knee_df = self._knee_points(pareto_df, f_cols, k=min(3, len(pareto_df)))

        return MOResult(
            pareto_df=pareto_df.reset_index(drop=True),
            top_df=top_df.reset_index(drop=True),
            knee_df=knee_df.reset_index(drop=True),
            x_cols=x_cols,
            f_cols=f_cols,
            g_cols=g_cols,
        )


# ============================== 다목적 빌드+런 헬퍼 ==============================

def make_multi_optimizer(
    artifacts: BuildArtifacts,
    use_catboost: bool = True,
    use_tabpfn: bool = False,
    safety_margin: float = 0.05,
    scaler: Optional[Any] = None,
) -> Dict[str, Optional[MultiObjectiveOptimizer]]:
    """다목적 옵티마이저 구성(타깃별 모델 매핑이 준비되어 있을 때만)."""
    out: Dict[str, Optional[MultiObjectiveOptimizer]] = {"mo_cat": None, "mo_tab": None}

    if use_catboost and artifacts.cat_models is not None:
        out["mo_cat"] = MultiObjectiveOptimizer(
            models=artifacts.cat_models,
            feature_names=artifacts.feature_names,
            important_features=artifacts.important_features,
            X_train=artifacts.df[artifacts.feature_names],
            scaler=scaler,
            safety_margin=safety_margin,
        )

    if use_tabpfn and artifacts.tabpfn_models is not None:
        out["mo_tab"] = MultiObjectiveOptimizer(
            models=artifacts.tabpfn_models,
            feature_names=artifacts.feature_names,
            important_features=artifacts.important_features,
            X_train=artifacts.df[artifacts.feature_names],
            scaler=scaler,
            safety_margin=safety_margin,
        )

    return out


def build_then_run_multi(
    build_cfg: BuildConfig,
    mo_run_cfg: RunConfigMO,
    for_optimal: pd.DataFrame,
    objective_names: Optional[List[str]] = None,
    use_catboost: bool = True,
    use_tabpfn: bool = False,
    scaler: Optional[Any] = None,
    safety_margin: float = 0.05,
) -> Dict[str, Any]:
    """다목적(NSGA-II) 빌드+런.

    Note:
        - build_cfg.targets 와 build_cfg.multi_target_specs 가 모두 필요.
        - for_optimal 은 N행 배치(All 모드) 입력.
    """
    artifacts = build_pipeline(build_cfg)

    if not build_cfg.targets:
        raise ValueError("build_cfg.targets must be provided for multi-objective run.")
    if not build_cfg.multi_target_specs:
        raise ValueError("build_cfg.multi_target_specs must be provided for multi-objective run.")

    optimizers = make_multi_optimizer(
        artifacts=artifacts,
        use_catboost=use_catboost,
        use_tabpfn=use_tabpfn,
        safety_margin=safety_margin,
        scaler=scaler,
    )

    results: Dict[str, Optional[MOResult]] = {"mo_cat": None, "mo_tab": None}

    if optimizers.get("mo_cat") is not None:
        results["mo_cat"] = optimizers["mo_cat"].optimize(
            for_optimal=for_optimal,
            target_specs=build_cfg.multi_target_specs,
            objective_names=objective_names or build_cfg.targets,
            pop_size=mo_run_cfg.pop_size,
            n_gen=mo_run_cfg.n_gen,
            random_seed=mo_run_cfg.random_seed,
            log_every=mo_run_cfg.log_every,
            top_n=mo_run_cfg.top_n,
        )

    if optimizers.get("mo_tab") is not None:
        results["mo_tab"] = optimizers["mo_tab"].optimize(
            for_optimal=for_optimal,
            target_specs=build_cfg.multi_target_specs,
            objective_names=objective_names or build_cfg.targets,
            pop_size=mo_run_cfg.pop_size,
            n_gen=mo_run_cfg.n_gen,
            random_seed=mo_run_cfg.random_seed,
            log_every=mo_run_cfg.log_every,
            top_n=mo_run_cfg.top_n,
        )

    return {"artifacts": artifacts, "results": results}


# ============================== 리포트 저장 유틸 (단일 목적 호환) ==============================

def save_optimization_report_csv(
    out: Dict[str, Any],
    for_optimal: pd.DataFrame,
    y: pd.Series,
    save_path: Path,
    func1: Callable[[pd.DataFrame], pd.DataFrame],
    func2: Callable[[pd.DataFrame], pd.DataFrame],
    important_features: Optional[List[str]] = None,
) -> Path:
    """(단일목적) 최적화 out 결과를 섹션별 CSV로 저장."""
    def _predict(model: Any, X: pd.DataFrame) -> np.ndarray:
        yhat = model.predict(X)
        return np.asarray(yhat).ravel()

    def _coerce_optimal_input(
        optimal_input: Any, feature_names: List[str], pref_feats: Optional[List[str]]
    ) -> Mapping[str, float]:
        if isinstance(optimal_input, Mapping):
            return {k: float(v) for k, v in optimal_input.items() if k in feature_names}
        if isinstance(optimal_input, Sequence) and not isinstance(optimal_input, (str, bytes)):
            seq = list(optimal_input)
            names = [c for c in (pref_feats or feature_names) if c in feature_names][: len(seq)]
            return {n: float(v) for n, v in zip(names, seq)}
        return {}

    def _apply_after(X: pd.DataFrame, ov_map: Mapping[str, float]) -> pd.DataFrame:
        if not ov_map:
            return X.copy()
        X2 = X.copy()
        for k, v in ov_map.items():
            if k in X2.columns:
                X2[k] = v
        return X2

    def _blank_row_like(cols: List[str]) -> pd.DataFrame:
        return pd.DataFrame([{c: "" for c in cols}])

    def _get_reference_X(model_obj: Any, fallback_df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
        for attr in ("X_train", "X_test"):
            Xc = getattr(model_obj, attr, None)
            if isinstance(Xc, pd.DataFrame) and len(Xc) > 0:
                return Xc[feature_names].copy()
        return fallback_df[feature_names].copy()

    def _optimal_vs_data_stats(ov_map: Mapping[str, float], ref_X: pd.DataFrame) -> pd.DataFrame:
        if not ov_map:
            return pd.DataFrame({"info": ["no optimal_input provided"]})
        cols = [c for c in ov_map.keys() if c in ref_X.columns]
        if not cols:
            return pd.DataFrame({"info": ["no matching features in reference data"]})
        stats = pd.DataFrame(
            {"min": ref_X[cols].min().astype(float),
             "mean": ref_X[cols].mean().astype(float),
             "max": ref_X[cols].max().astype(float)}
        ).T
        optimal_row = pd.DataFrame([ov_map], index=["optimal"])[cols].astype(float)
        return pd.concat([stats, optimal_row], axis=0)

    artifacts: Dict[str, Any] = out.get("artifacts", {})
    cat_model = artifacts.get("cat_model_obj", None)
    tab_model = artifacts.get("tabpfn_model_obj", None)
    feature_names: List[str] = artifacts.get("feature_names", list(for_optimal.columns))
    artifacts_df = artifacts.get("df", for_optimal)

    X_base = for_optimal[feature_names].copy()
    prdt = np.asarray(y).ravel()

    sections = [
        ("bo_cat", cat_model),
        ("bo_tab", tab_model),
        ("ga_cat", cat_model),
        ("ga_tab", tab_model),
    ]

    blocks: List[pd.DataFrame] = []

    for key, model in sections:
        res = out.get("results", {}).get(key, None)
        if res is None or model is None:
            continue

        best_x = res[0]
        y_before = _predict(model, X_base)
        ov = _coerce_optimal_input(best_x, feature_names, important_features)
        X_after = _apply_after(X_base, ov)
        y_after = _predict(model, X_after)

        head = pd.DataFrame({"section": [f"[SECTION] {key}"]})

        core_df = pd.DataFrame(
            {
                "idx": for_optimal.index,
                "prdt": prdt,
                "before_optim": y_before,
                "after_optim": y_after,
            }
        )

        f1 = func1(core_df[["prdt", "before_optim", "after_optim"]])
        f2 = func2(core_df[["prdt", "before_optim", "after_optim"]])

        ref_X = _get_reference_X(model, artifacts_df, feature_names)
        stats_df = _optimal_vs_data_stats(ov, ref_X).reset_index().rename(columns={"index": "stat"})

        blocks.extend(
            [
                head,
                core_df,
                _blank_row_like(core_df.columns.tolist()),
                f1,
                _blank_row_like(f1.columns.tolist()),
                f2,
                _blank_row_like(f2.columns.tolist()),
                stats_df,
                pd.DataFrame([{"": ""}]),
            ]
        )

    save_path = Path(save_path).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if not blocks:
        pd.DataFrame([{"info": "no active results to save"}]).to_csv(save_path, index=False)
        return save_path

    pd.concat(blocks, ignore_index=True).to_csv(save_path, index=False)
    return save_path
