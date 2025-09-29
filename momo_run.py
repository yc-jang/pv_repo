# run.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Mapping, Sequence, Union
import os
import pickle
import pandas as pd
import numpy as np

# ---- 외부 구현(기존) ----
from bayesian_optimizer import BayesianOptimzation
from genetic_optimizer import GeneticOptimizer
from catboost_runner import CatBoostRunner
from tabpfn_runner import TabPFNRunner

# ---- 다목적 최적화(새 모듈) ----
from multi_objective_optimizer import (
    TargetSpec,
    RunConfigMO,
    MOResult,
    MultiObjectiveOptimizer,
)

# =============================================================================
# 공통 유틸
# =============================================================================

def to_relpath(path: Path, start: Optional[Path] = None) -> str:
    base = Path.cwd() if start is None else Path(start)
    try:
        return os.path.relpath(Path(path).resolve(), start=base.resolve())
    except ValueError:
        return str(Path(path).resolve())


def load_pickle_df(pickle_path: Path) -> pd.DataFrame:
    if not pickle_path.exists():
        raise FileNotFoundError(f"pickle not found: {pickle_path}")
    with open(pickle_path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, pd.DataFrame):
        raise ValueError(f"Loaded object is not a DataFrame: type={type(obj)}")
    return obj


def _extract_model_object(runner_or_none: Optional[Any]) -> Optional[Any]:
    if runner_or_none is None:
        return None
    return getattr(runner_or_none, "model", runner_or_none)


def _infer_feature_names(
    preferred: Optional[List[str]],
    cat_runner: Optional[CatBoostRunner],
    df: Optional[pd.DataFrame],
) -> List[str]:
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
    raise ValueError("Unable to infer feature_names; please provide feature_names explicitly.")


# =============================================================================
# 단일 목적(SO) 구성
# =============================================================================

# ---------- 설정 ----------
@dataclass(frozen=True)
class BuildConfigSO:
    """단일 목적 빌드 설정."""
    base_dir: Path
    data_pickle: Path
    target: str
    column_sets: Dict[str, List[str]]
    column_set_key: str
    desired_value: float
    feature_names: Optional[List[str]] = None


@dataclass(frozen=True)
class RunConfigBO:
    """베이지안 최적화 실행 설정."""
    init_points: int
    n_iter: int


@dataclass(frozen=True)
class RunConfigGA:
    """유전 알고리즘 실행 설정."""
    population: int
    parents: int
    generations: int


@dataclass(frozen=True)
class ArtifactsSO:
    """단일 목적 빌드 산출물."""
    df: pd.DataFrame
    cat_model_obj: Optional[Any]
    tabpfn_model_obj: Optional[Any]
    feature_names: List[str]
    important_features: List[str]
    catboost_model_path: Optional[str]
    tabpfn_ckpt_path: Optional[str]


# ---------- 로더 ----------
def find_model_paths(models_dir: Path, target: str) -> Tuple[Optional[Path], Optional[Path]]:
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


# ---------- 빌드 ----------
def build_pipeline_so(cfg: BuildConfigSO) -> ArtifactsSO:
    base_dir = cfg.base_dir.resolve()
    models_dir = (base_dir / "models").resolve()
    df = load_pickle_df(cfg.data_pickle.resolve())

    # 최신 모델 경로
    cat_path, tab_path = find_model_paths(models_dir, cfg.target)
    cat_runner = CatBoostRunner().load_model(cat_path) if cat_path else None
    tab_runner = TabPFNRunner().load_model(tab_path) if tab_path else None

    feature_names = _infer_feature_names(cfg.feature_names, cat_runner, df)

    if cfg.column_set_key not in cfg.column_sets:
        raise KeyError(f"column_sets does not contain key='{cfg.column_set_key}'")
    important_features = cfg.column_sets[cfg.column_set_key]
    if not isinstance(important_features, list):
        raise ValueError(f"column_sets['{cfg.column_set_key}'] must be a list of column names")

    return ArtifactsSO(
        df=df,
        cat_model_obj=_extract_model_object(cat_runner),
        tabpfn_model_obj=_extract_model_object(tab_runner),
        feature_names=feature_names,
        important_features=important_features,
        catboost_model_path=to_relpath(cat_path) if cat_path else None,
        tabpfn_ckpt_path=to_relpath(tab_path) if tab_path else None,
    )


# ---------- SO Optimizer Pack ----------
@dataclass
class OptimizerPackSO:
    bo_cat: Optional[BayesianOptimzation] = None
    bo_tab: Optional[BayesianOptimzation] = None
    ga_cat: Optional[GeneticOptimizer] = None
    ga_tab: Optional[GeneticOptimizer] = None


def make_optimizers_so(
    artifacts: ArtifactsSO,
    cfg: BuildConfigSO,
    use_catboost: bool = True,
    use_tabpfn: bool = False,
    make_bayes: bool = True,
    make_genetic: bool = True,
) -> OptimizerPackSO:
    pack = OptimizerPackSO()
    if use_catboost and artifacts.cat_model_obj is not None:
        if make_bayes:
            pack.bo_cat = BayesianOptimzation(
                model=artifacts.cat_model_obj,
                model_type="catboost",
                feature_names=artifacts.feature_names,
                desired_value=cfg.desired_value,
                important_feature=artifacts.important_features,
            )
        if make_genetic:
            pack.ga_cat = GeneticOptimizer(
                model=artifacts.cat_model_obj,
                model_type="catboost",
                feature_names=artifacts.feature_names,
                desired_value=cfg.desired_value,
                important_feature=artifacts.important_features,
            )
    if use_tabpfn and artifacts.tabpfn_model_obj is not None:
        if make_bayes:
            pack.bo_tab = BayesianOptimzation(
                model=artifacts.tabpfn_model_obj,
                model_type="tabpfn",
                feature_names=artifacts.feature_names,
                desired_value=cfg.desired_value,
                important_feature=artifacts.important_features,
            )
        if make_genetic:
            pack.ga_tab = GeneticOptimizer(
                model=artifacts.tabpfn_model_obj,
                model_type="tabpfn",
                feature_names=artifacts.feature_names,
                desired_value=cfg.desired_value,
                important_feature=artifacts.important_features,
            )
    return pack


# ---------- SO 실행 ----------
def run_bayes(optimizer: BayesianOptimzation, run_cfg: RunConfigBO) -> Tuple[Any, float]:
    return optimizer.optimize(init_points=run_cfg.init_points, n_iter=run_cfg.n_iter)


def run_genetic(optimizer: GeneticOptimizer, run_cfg: RunConfigGA) -> Tuple[Any, float]:
    return optimizer.optimize(
        population=run_cfg.population, parents=run_cfg.parents, generations=run_cfg.generations
    )


def build_then_run_so(
    build_cfg: BuildConfigSO,
    bo_run_cfg: Optional[RunConfigBO] = None,
    ga_run_cfg: Optional[RunConfigGA] = None,
    use_catboost: bool = True,
    use_tabpfn: bool = False,
    do_bayes: bool = True,
    do_genetic: bool = True,
) -> Dict[str, Any]:
    artifacts = build_pipeline_so(build_cfg)
    pack = make_optimizers_so(
        artifacts=artifacts,
        cfg=build_cfg,
        use_catboost=use_catboost,
        use_tabpfn=use_tabpfn,
        make_bayes=do_bayes,
        make_genetic=do_genetic,
    )

    results: Dict[str, Optional[Tuple[Any, float]]] = {
        "bo_cat": None, "bo_tab": None, "ga_cat": None, "ga_tab": None
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

    return {"mode": "single", "artifacts": artifacts, "results": results}


# =============================================================================
# 다목적(MO) 구성
# =============================================================================

# ---------- 설정 ----------
@dataclass(frozen=True)
class BuildConfigMO:
    """다목적 빌드 설정."""
    base_dir: Path
    data_pickle: Path
    targets: List[str]                       # 타깃명 리스트(파일명에 포함된다고 가정)
    column_sets: Dict[str, List[str]]
    column_set_key: str
    feature_names: Optional[List[str]] = None


@dataclass(frozen=True)
class ArtifactsMO:
    """다목적 빌드 산출물."""
    df: pd.DataFrame
    cat_models: Optional[Dict[str, Any]]
    tabpfn_models: Optional[Dict[str, Any]]
    feature_names: List[str]
    important_features: List[str]
    catboost_model_paths: Optional[Dict[str, str]]
    tabpfn_ckpt_paths: Optional[Dict[str, str]]


# ---------- 로더 ----------
def find_models_by_targets(models_dir: Path, targets: List[str]) -> Tuple[Dict[str, Optional[Path]], Dict[str, Optional[Path]]]:
    cat_map: Dict[str, Optional[Path]] = {}
    tab_map: Dict[str, Optional[Path]] = {}
    for tgt in targets:
        cat_map[tgt], tab_map[tgt] = find_model_paths(models_dir, target=tgt)
    return cat_map, tab_map


# ---------- 빌드 ----------
def build_pipeline_mo(
    cfg: BuildConfigMO,
    feature_probe_target: Optional[str] = None,
) -> ArtifactsMO:
    """다목적 빌드. feature_names 추정은 probe 타깃(없으면 첫 타깃)의 모델에서 시도."""
    base_dir = cfg.base_dir.resolve()
    models_dir = (base_dir / "models").resolve()
    df = load_pickle_df(cfg.data_pickle.resolve())

    # 타깃별 최신 경로 수집
    cat_paths, tab_paths = find_models_by_targets(models_dir, cfg.targets)

    # feature_names 추정용 probe
    probe = feature_probe_target or (cfg.targets[0] if cfg.targets else None)
    cat_runner_probe = None
    if probe and cat_paths.get(probe) is not None:
        cat_runner_probe = CatBoostRunner().load_model(cat_paths[probe])

    feature_names = _infer_feature_names(cfg.feature_names, cat_runner_probe, df)

    if cfg.column_set_key not in cfg.column_sets:
        raise KeyError(f"column_sets does not contain key='{cfg.column_set_key}'")
    important_features = cfg.column_sets[cfg.column_set_key]
    if not isinstance(important_features, list):
        raise ValueError(f"column_sets['{cfg.column_set_key}'] must be a list of column names")

    # 타깃별 로드
    cat_models: Dict[str, Any] = {}
    tab_models: Dict[str, Any] = {}
    cat_paths_map: Dict[str, str] = {}
    tab_paths_map: Dict[str, str] = {}
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

    if all(v is None for v in cat_models.values()):
        cat_models = None  # type: ignore
    if all(v is None for v in tab_models.values()):
        tab_models = None  # type: ignore
    if not cat_paths_map:
        cat_paths_map = None  # type: ignore
    if not tab_paths_map:
        tab_paths_map = None  # type: ignore

    return ArtifactsMO(
        df=df,
        cat_models=cat_models if cat_models else None,
        tabpfn_models=tab_models if tab_models else None,
        feature_names=feature_names,
        important_features=important_features,
        catboost_model_paths=cat_paths_map if cat_paths_map else None,
        tabpfn_ckpt_paths=tab_paths_map if tab_paths_map else None,
    )


# ---------- MO Optimizer ----------
def make_optimizer_mo(
    artifacts: ArtifactsMO,
    use_catboost: bool = True,
    use_tabpfn: bool = False,
    safety_margin: float = 0.05,
    scaler: Optional[Any] = None,
) -> Dict[str, Optional[MultiObjectiveOptimizer]]:
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


# ---------- MO 실행 ----------
def build_then_run_mo(
    build_cfg: BuildConfigMO,
    run_cfg: RunConfigMO,
    for_optimal: pd.DataFrame,
    target_specs: Dict[str, TargetSpec],
    objective_names: Optional[List[str]] = None,
    use_catboost: bool = True,
    use_tabpfn: bool = False,
    scaler: Optional[Any] = None,
    safety_margin: float = 0.05,
) -> Dict[str, Any]:
    artifacts = build_pipeline_mo(build_cfg)
    if use_catboost and artifacts.cat_models is None and use_tabpfn and artifacts.tabpfn_models is None:
        raise ValueError("No models available for multi-objective run.")

    optimizers = make_optimizer_mo(
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
            target_specs=target_specs,
            objective_names=objective_names or build_cfg.targets,
            pop_size=run_cfg.pop_size,
            n_gen=run_cfg.n_gen,
            random_seed=run_cfg.random_seed,
            log_every=run_cfg.log_every,
            top_n=run_cfg.top_n,
        )
    if optimizers.get("mo_tab") is not None:
        results["mo_tab"] = optimizers["mo_tab"].optimize(
            for_optimal=for_optimal,
            target_specs=target_specs,
            objective_names=objective_names or build_cfg.targets,
            pop_size=run_cfg.pop_size,
            n_gen=run_cfg.n_gen,
            random_seed=run_cfg.random_seed,
            log_every=run_cfg.log_every,
            top_n=run_cfg.top_n,
        )

    return {"mode": "multi", "artifacts": artifacts, "results": results}


# =============================================================================
# (선택) 리포트 유틸 - 단일 목적 호환
# =============================================================================

def save_optimization_report_csv(
    out: Dict[str, Any],
    for_optimal: pd.DataFrame,
    y: pd.Series,
    save_path: Path,
    func1: Callable[[pd.DataFrame], pd.DataFrame],
    func2: Callable[[pd.DataFrame], pd.DataFrame],
    important_features: Optional[List[str]] = None,
) -> Path:
    """(단일 목적) 섹션별 CSV 저장."""
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
    cat_model = getattr(artifacts, "cat_model_obj", None) or artifacts.get("cat_model_obj", None)  # ArtifactsSO dict/obj 모두 대응
    tab_model = getattr(artifacts, "tabpfn_model_obj", None) or artifacts.get("tabpfn_model_obj", None)
    feature_names: List[str] = getattr(artifacts, "feature_names", None) or artifacts.get("feature_names", list(for_optimal.columns))  # type: ignore
    artifacts_df = getattr(artifacts, "df", None) or artifacts.get("df", for_optimal)  # type: ignore

    X_base = for_optimal[feature_names].copy()
    prdt = np.asarray(y).ravel()

    sections = [("bo_cat", cat_model), ("bo_tab", tab_model), ("ga_cat", cat_model), ("ga_tab", tab_model)]
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
        core_df = pd.DataFrame({"idx": for_optimal.index, "prdt": prdt, "before_optim": y_before, "after_optim": y_after})
        f1 = func1(core_df[["prdt", "before_optim", "after_optim"]])
        f2 = func2(core_df[["prdt", "before_optim", "after_optim"]])

        ref_X = _get_reference_X(model, artifacts_df, feature_names)
        stats_df = _optimal_vs_data_stats(ov, ref_X).reset_index().rename(columns={"index": "stat"})

        blocks.extend([head, core_df, _blank_row_like(core_df.columns.tolist()), f1, _blank_row_like(f1.columns.tolist()),
                       f2, _blank_row_like(f2.columns.tolist()), stats_df, pd.DataFrame([{"": ""}])])

    save_path = Path(save_path).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if not blocks:
        pd.DataFrame([{"info": "no active results to save"}]).to_csv(save_path, index=False)
        return save_path

    pd.concat(blocks, ignore_index=True).to_csv(save_path, index=False)
    return save_path


# =============================================================================
# CSV 파라미터 기반 실행(미래 확장용 스텁)
# =============================================================================

@dataclass(frozen=True)
class CsvRuntimeParams:
    """CSV에서 읽을 런타임 파라미터(미래 확장용).
    - mode: 'single' 또는 'multi'
    - optimizer: 'bo'/'ga' (single), 'nsga2' (multi)
    - targets / desired / bounds / specs 등은 컬럼 규약에 맞춰 해석
    """
    mode: str
    raw: pd.DataFrame


def load_runtime_params_from_csv(path: Path) -> CsvRuntimeParams:
    """CSV를 로드해 런타임 파라미터 프레임을 반환(파싱은 프로젝트 규약에 맞춰 후속 구현)."""
    df = pd.read_csv(path)
    # 간단한 힌트만 남기고 구체 파싱은 v2에서: df 컬럼 규약 확정 후 작성
    mode = df.attrs.get("mode", "single") if hasattr(df, "attrs") else "single"
    return CsvRuntimeParams(mode=mode, raw=df)
