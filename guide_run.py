from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os
import pickle
import pandas as pd

# 외부 구현 클래스(실제 모듈/클래스가 존재한다고 가정)
from bayesian_optimizer import BayesianOptimzation
from genetic_optimizer import GeneticOptimizer
from catboost_runner import CatBoostRunner
from tabpfn_runner import TabPFNRunner


# ============================== 설정/유틸 ==============================

@dataclass(frozen=True)
class BuildConfig:
    """파이프라인 빌드(준비) 단계 설정.

    Args:
        base_dir: 프로젝트 루트 디렉터리(예: Path(__file__).parent).
        data_pickle: end2end 데이터 피클 경로.
        target: CatBoost 모델 선택에 사용할 TARGET 문자열(파일명 부분 일치).
        column_sets: {'only_cond':[...], 'matr_cond':[...]} 형태의 컬럼셋.
        column_set_key: important_feature를 고를 키(예: 'only_cond' 또는 'matr_cond').
        desired_value: 목표 스칼라 값(예: TARGET 규격값의 평균 등).
        feature_names: (선택) 학습에 사용한 전체 피처 리스트.
                       미지정 시 모델/데이터로부터 추정 시도.
    """
    base_dir: Path
    data_pickle: Path
    target: str
    column_sets: Dict[str, List[str]]
    column_set_key: str
    desired_value: float
    feature_names: Optional[List[str]] = None


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
class BuildArtifacts:
    """빌드 단계에서 준비된 산출물.

    Attributes:
        df: 로드된 전체 DataFrame.
        cat_model_obj: CatBoostRunner().load_model(...) 이후 내부 모델 객체(가능하면 .model, 없으면 러너 인스턴스).
        tabpfn_model_obj: TabPFNRunner().load_model(...) 이후 내부 모델 객체(가능하면 .model, 없으면 러너 인스턴스).
        feature_names: 학습에 사용한 전체 피처 리스트.
        important_features: 선택된 column_set_key에 해당하는 중요 컬럼 리스트.
        catboost_model_path: 선택된 CatBoost .pkl 상대경로(없으면 None).
        tabpfn_ckpt_path: 선택된 TabPFN .ckpt 상대경로(없으면 None).
    """
    df: pd.DataFrame
    cat_model_obj: Optional[Any]
    tabpfn_model_obj: Optional[Any]
    feature_names: List[str]
    important_features: List[str]
    catboost_model_path: Optional[str]
    tabpfn_ckpt_path: Optional[str]


def to_relpath(path: Path, start: Optional[Path] = None) -> str:
    """Path를 OS 상대 경로 문자열로 변환.

    드라이브가 다르거나 기준 불가 시 절대경로로 반환.
    """
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

    # 재귀 탐색으로 후보 수집
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


def _infer_feature_names(
    preferred: Optional[List[str]],
    cat_runner: Optional[CatBoostRunner],
    df: Optional[pd.DataFrame],
) -> List[str]:
    """feature_names 추정: 우선순위 = preferred → cat_runner 내부 정보 → df.columns.

    Note:
        - 구현체에 따라 cat_runner.model.feature_names_ 또는 cat_runner.model.feature_names
          혹은 cat_runner.X_train/X_test.columns 등 다양한 케이스가 가능. 여기서는
          대표적인 속성만 가볍게 시도하고 실패 시 df.columns로 폴백.
    """
    if preferred is not None:
        return list(preferred)

    # CatBoostRunner 기반 추정 시도
    if cat_runner is not None:
        model_obj = getattr(cat_runner, "model", None)
        if model_obj is not None:
            for attr in ("feature_names_", "feature_names"):
                names = getattr(model_obj, attr, None)
                if names is not None:
                    return list(names)
        # 추가 힌트: 러너가 X_train/X_test를 보유할 수 있음
        for attr in ("X_train", "X_test"):
            x_obj = getattr(cat_runner, attr, None)
            if isinstance(x_obj, pd.DataFrame):
                return x_obj.columns.tolist()

    # 마지막 폴백: df.columns
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
    """데이터/모델/피처/중요컬럼을 준비하여 런 전 상태로 만든다.

    Args:
        cfg: BuildConfig

    Returns:
        BuildArtifacts

    Raises:
        KeyError: column_set_key가 column_sets에 없을 때.
        FileNotFoundError/ValueError: 데이터 로딩 실패 등.
    """
    base_dir = cfg.base_dir.resolve()
    models_dir = (base_dir / "models").resolve()
    data_pickle = cfg.data_pickle.resolve()

    # 1) 데이터 로드
    df = load_pickle_df(data_pickle)

    # 2) 모델 경로 탐색 후 로드
    cat_path, tab_path = find_model_paths(models_dir=models_dir, target=cfg.target)

    cat_runner: Optional[CatBoostRunner] = None
    tab_runner: Optional[TabPFNRunner] = None

    if cat_path is not None:
        cat_runner = CatBoostRunner().load_model(cat_path)
    if tab_path is not None:
        tab_runner = TabPFNRunner().load_model(tab_path)

    # 3) feature_names 결정
    feature_names = _infer_feature_names(cfg.feature_names, cat_runner, df)

    # 4) important_features 결정
    if cfg.column_set_key not in cfg.column_sets:
        raise KeyError(f"column_sets does not contain key='{cfg.column_set_key}'")
    important_features = cfg.column_sets[cfg.column_set_key]
    if not isinstance(important_features, list):
        raise ValueError(f"column_sets['{cfg.column_set_key}'] must be a list of column names")

    # 5) 모델 오브젝트 추출(.model 있으면 추출)
    cat_model_obj = _extract_model_object(cat_runner)
    tabpfn_model_obj = _extract_model_object(tab_runner)

    return BuildArtifacts(
        df=df,
        cat_model_obj=cat_model_obj,
        tabpfn_model_obj=tabpfn_model_obj,
        feature_names=feature_names,
        important_features=important_features,
        catboost_model_path=to_relpath(cat_path) if cat_path else None,
        tabpfn_ckpt_path=to_relpath(tab_path) if tab_path else None,
    )


# ============================== 런 단계 ==============================

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
    """빌드 산출물로부터 옵티마이저 인스턴스들을 구성한다.

    Note:
        각 옵티마이저 초기화는 다음과 같은 공통 형식을 따른다.
        GeneticOptimizer(
            model=<model object>,
            model_type='catboost' 또는 'tabpfn',
            feature_names=<List[str]>,
            desired_value=<float>,
            important_feature=<List[str]>
        )
        BayesianOptimzation(...) 도 동일한 서명으로 초기화한다고 가정.
    """
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
    """베이지안 최적화 실행 (호출 시점 파라미터)."""
    return optimizer.optimize(init_points=run_cfg.init_points, n_iter=run_cfg.n_iter)


def run_genetic(optimizer: GeneticOptimizer, run_cfg: RunConfigGA) -> Tuple[Any, float]:
    """유전 알고리즘 실행 (호출 시점 파라미터)."""
    return optimizer.optimize(
        population=run_cfg.population,
        parents=run_cfg.parents,
        generations=run_cfg.generations,
    )


# ============================== 파이프라인 헬퍼 ==============================

def build_then_run(
    build_cfg: BuildConfig,
    bo_run_cfg: Optional[RunConfigBO] = None,
    ga_run_cfg: Optional[RunConfigGA] = None,
    use_catboost: bool = True,
    use_tabpfn: bool = False,
    do_bayes: bool = True,
    do_genetic: bool = True,
) -> Dict[str, Any]:
    """한 번에 빌드하고 원하는 조합으로 런까지 수행.

    Args:
        build_cfg: 빌드 설정.
        bo_run_cfg: BO 실행 설정(선택).
        ga_run_cfg: GA 실행 설정(선택).
        use_catboost: CatBoost 기반 최적화 포함 여부.
        use_tabpfn: TabPFN  기반 최적화 포함 여부.
        do_bayes: 베이지안 최적화 수행 여부.
        do_genetic: 유전 최적화 수행 여부.

    Returns:
        {
          "artifacts": BuildArtifacts,
          "results": {
              "bo_cat": (x, y) | None,
              "bo_tab": (x, y) | None,
              "ga_cat": (x, y) | None,
              "ga_tab": (x, y) | None,
          }
        }
    """
    artifacts = build_pipeline(build_cfg)
    pack = make_optimizers(
        artifacts=artifacts,
        build_cfg=build_cfg,
        use_catboost=use_catboost,
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


# ============================== 사용 예시 ==============================
# base = Path(__file__).resolve().parent
# build_cfg = BuildConfig(
#     base_dir=base,
#     data_pickle=base / "data" / "end2end_data.pkl",
#     target="YOUR_TARGET",                    # CatBoost .pkl 파일명에 포함될 문자열
#     column_sets={"only_cond": [...], "matr_cond": [...]},
#     column_set_key="only_cond",
#     desired_value=0.85,                      # 예: TARGET 규격 평균
#     feature_names=None,                      # 있으면 제공, 없으면 추정
# )
#
# bo_cfg = RunConfigBO(init_points=8, n_iter=32)
# ga_cfg = RunConfigGA(population=60, parents=12, generations=40)
#
# out = build_then_run(
#     build_cfg=build_cfg,
#     bo_run_cfg=bo_cfg,
#     ga_run_cfg=ga_cfg,
#     use_catboost=True,
#     use_tabpfn=False,   # TabPFN까지 동시에 돌리려면 True
#     do_bayes=True,
#     do_genetic=True,
# )
# # out["artifacts"].feature_names, important_features, 모델 경로 등 접근 가능
# # out["results"]["bo_cat"] → (best_x, best_y) 형태
