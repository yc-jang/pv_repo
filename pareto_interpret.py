from __future__ import annotations
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd

from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_score

import plotly.express as px
import plotly.graph_objects as go


class GlobalImportanceExplainer:
    """전역 중요도(누가 지배?) 전용 클래스.
    
    - fit(): 전처리 + 목적별 파이프라인 학습 + permutation importance(PI)
    - importance_: 원본 컬럼 × 목적 중요도표(+ avg_importance, 정렬)
    - plot_*(): Heatmap/Bar 시각화
    - quality_report(): 각 목적별 CV R²(간이)로 해석 전제 검증
    - stability_check(): PI 반복/시드 변동 시 Top-K 안정성(겹침률) 점검
    """

    def __init__(
        self,
        n_estimators: int = 400,
        n_repeats: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        # 모델/PI 설정값 저장
        self.n_estimators = n_estimators
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.n_jobs = n_jobs

        # 학습 산출물 저장소
        self.models_: Dict[str, Pipeline] = {}
        self.importance_: Optional[pd.DataFrame] = None
        self.feature_types_: Optional[Tuple[List[str], List[str]]] = None  # (num_cols, cat_cols)
        self.index_: Optional[pd.Index] = None

        # 로깅(요청 지침 포맷에 맞춰 no-op 기본)
        logger.remove()
        logger.add(
            sink=lambda msg: None,
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} {level} [{name}:{line}] {message}",
        )

    # -------------------------------
    # 내부 유틸: 전처리 구성 & 역매핑
    # -------------------------------
    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """숫자=median, 범주=most_frequent+OHE(handle_unknown) 전처리 구성."""
        # (1) 숫자/범주 자동 분리 — 수치형 판단 기준은 np.number
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.columns.difference(num_cols).tolist()
        self.feature_types_ = (num_cols, cat_cols)

        # (2) 결측 대체 파이프 — 트리 모델에 스케일링 불필요
        num_pipe = Pipeline([("imp", SimpleImputer(strategy="median"))])
        cat_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]) if len(cat_cols) else None

        # (3) ColumnTransformer 구성
        transformers = []
        if len(num_cols):
            transformers.append(("num", num_pipe, num_cols))
        if cat_pipe is not None:
            transformers.append(("cat", cat_pipe, cat_cols))

        pre = ColumnTransformer(transformers=transformers, remainder="drop", n_jobs=self.n_jobs)
        return pre

    @staticmethod
    def _origin_map(pre: ColumnTransformer) -> pd.Series:
        """전처리 후 특성명 → 원본 컬럼명 매핑(원-핫 파생을 원 컬럼으로 집계하기 위함)."""
        names_after, origin = [], []
        for name, trans, cols in pre.transformers_:
            if name == "num":
                for c in cols:
                    names_after.append(c)            # 수치 특성은 1:1
                    origin.append(c)
            elif name == "cat":
                oh: OneHotEncoder = trans.named_steps["oh"]
                oh_names = oh.get_feature_names_out(cols).tolist()  # 예: "col_A", "col_B", ...
                names_after.extend(oh_names)
                origin.extend([s.split("_", 1)[0] for s in oh_names])  # 원 컬럼명으로 복원
        return pd.Series(origin, index=names_after, name="origin")

    def _make_pipeline(self, pre: ColumnTransformer) -> Pipeline:
        """전처리 + RF 회귀 파이프라인 구성."""
        rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            oob_score=False,
        )
        return Pipeline([("pre", pre), ("rf", rf)])

    # -------------------------------
    # 핵심: 학습 + PI 계산 (한 줄 한 줄 주석)
    # -------------------------------
    def fit(self, X: pd.DataFrame, F: pd.DataFrame) -> "GlobalImportanceExplainer":
        """X(조작조건)→F(목적) 전역 중요도 학습/계산."""
        # 1) 원본 보존 + 인덱스 정렬로 행 순서 일치 확보(누락/뒤섞임 방지)
        X = X.copy().sort_index()
        F = F.copy().sort_index()

        # 2) 공통 인덱스 교집합만 채택 (X/F 행 어긋남으로 인한 데이터 스누핑 방지)
        idx = X.index.intersection(F.index)
        X, F = X.loc[idx], F.loc[idx]
        self.index_ = idx

        # 3) 목적값 결측 행 제거 (학습 불가) — 목적 어느 하나라도 NaN이면 드롭
        ok = ~F.isna().any(axis=1)
        if not ok.all():
            X, F = X.loc[ok], F.loc[ok]

        # 4) 전처리 파이프라인 구성(숫자/범주 분리, 필수 결측 대체, 범주 OHE)
        pre = self._build_preprocessor(X)

        # 5) 목적별 파이프라인 학습 및 permutation importance 계산을 위한 준비
        self.models_.clear()
        frames: List[pd.DataFrame] = []

        # 6) 목적별 루프 — 다목적을 “목적별 단일모델”로 학습(해석/PI 산정이 명확해짐)
        for tgt in F.columns:
            y = F[tgt].to_numpy()  # 6-1) 현재 타깃의 벡터 추출

            pipe = self._make_pipeline(pre)  # 6-2) 동일 전처리 공유 파이프라인
            pipe.fit(X, y)                   # 6-3) 학습(전처리 후 RF)

            self.models_[tgt] = pipe         # 6-4) 추후 재사용(PDP/검증 등)

            # 6-5) 전처리 결과의 파생 특성명을 원 컬럼으로 역매핑하기 위한 테이블
            pre_fitted: ColumnTransformer = pipe.named_steps["pre"]
            origin = self._origin_map(pre_fitted)

            # 6-6) 비모수적 전역 중요도: permutation importance
            #      - 전처리 포함 파이프라인 전체를 대상으로, 입력 특징(전처리 후) 셔플 → 성능 저하량 평균
            #      - 목적별로 합=1로 정규화하여 목적 간 상대비교 용이
            pi = permutation_importance(
                estimator=pipe,
                X=X,
                y=y,
                n_repeats=self.n_repeats,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
            imp_mean = pd.Series(pi.importances_mean, index=origin.index)

            # 6-7) 원-핫으로 분해된 파생 중요도를 “원본 컬럼 단위”로 합산 → 인간 해석 가능 형태
            imp_by_origin = imp_mean.groupby(origin).sum()

            # 6-8) 목적별로 합=1 정규화(0 나눗셈 방지)
            total = imp_by_origin.sum()
            s = imp_by_origin / total if total > 0 else imp_by_origin

            # 6-9) 목적 이름으로 시리즈 라벨링 후 수집
            frames.append(s.rename(tgt))

        # 7) 목적별 중요도 시리즈를 컬럼 방향으로 병합, 없는 곳은 0으로 채움
        imp = pd.concat(frames, axis=1).fillna(0.0)

        # 8) 평균 중요도 컬럼 추가(전략 변수 우선순위 파악) 후 중요도 순으로 정렬
        imp["avg_importance"] = imp.mean(axis=1)
        imp = imp.sort_values("avg_importance", ascending=False)

        # 9) 산출물 저장
        self.importance_ = imp
        return self

    # -------------------------------
    # 전역 중요도 시각화
    # -------------------------------
    def plot_heatmap(self, top_k: int = 25) -> go.Figure:
        """상위 top_k 원본 특성 × 목적 중요도 히트맵."""
        assert self.importance_ is not None, "Call fit() first."
        cols_targets = [c for c in self.importance_.columns if c != "avg_importance"]
        df_ = self.importance_.head(top_k).copy()
        fig = px.imshow(
            df_[cols_targets],
            labels=dict(x="Targets", y="Features", color="Importance"),
            x=cols_targets, y=df_.index,
            aspect="auto", color_continuous_scale="Viridis", origin="upper",
        )
        fig.update_layout(title=f"Feature Importance Heatmap (Top {top_k})")
        return fig

    def plot_bar(self, top_k: int = 20) -> go.Figure:
        """평균 중요도 막대(전략 변수 우선순위)."""
        assert self.importance_ is not None, "Call fit() first."
        s = self.importance_["avg_importance"].head(top_k)
        fig = go.Figure(go.Bar(x=s.index.tolist(), y=s.values.tolist()))
        fig.update_layout(
            title=f"Average Importance (Top {top_k})",
            xaxis_title="Features", yaxis_title="Avg Importance", xaxis_tickangle=-30
        )
        return fig

    # -------------------------------
    # 품질/안정성 보조 리포트(선택)
    # -------------------------------
    def quality_report(self, X: pd.DataFrame, F: pd.DataFrame, cv: int = 5) -> pd.Series:
        """간이 품질 점검: 목적별 파이프라인으로 KFold R² 평균을 산출.
        - 해석의 전제(모델이 최소한 의미 있는가)를 확인하기 위함.
        """
        assert self.models_, "Call fit() first."
        # 인덱스 정합
        X = X.loc[self.index_]
        F = F.loc[self.index_]

        scores = {}
        kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        for tgt, pipe in self.models_.items():
            y = F[tgt].to_numpy()
            # cross_val_score가 Pipeline에 대해 전처리 포함 평가
            sc = cross_val_score(pipe, X, y, scoring="r2", cv=kf, n_jobs=self.n_jobs)
            scores[tgt] = float(np.mean(sc))
        return pd.Series(scores, name="cv_r2")

    def stability_check(
        self, X: pd.DataFrame, F: pd.DataFrame, top_k: int = 15, seeds: List[int] = [101, 202, 303]
    ) -> pd.DataFrame:
        """PI 재현성 점검: 시드를 바꿔 재계산한 상위 Top-K 겹침률(%) 리포트."""
        assert self.importance_ is not None, "Call fit() first."
        base_top = set(self.importance_.head(top_k).index)
        rows = []
        for sd in seeds:
            # 임시 explainer로 동일 파이프라인 재적합(시드만 변경)
            tmp = GlobalImportanceExplainer(
                n_estimators=self.n_estimators,
                n_repeats=self.n_repeats,
                random_state=sd,
                n_jobs=self.n_jobs,
            ).fit(X, F)
            cand = set(tmp.importance_.head(top_k).index)  # type: ignore
            overlap = 100.0 * len(base_top & cand) / top_k
            rows.append({"seed": sd, "top_k": top_k, "overlap_pct": overlap})
        return pd.DataFrame(rows)

    def corr_heatmap(self, X: pd.DataFrame, top_k: int = 20) -> go.Figure:
        """상위 중요 변수들 간의 상관(숫자형만) 히트맵 — 공선성/해석 분산 보조."""
        assert self.importance_ is not None, "Call fit() first."
        top_feats = [c for c in self.importance_.head(top_k).index if c in X.columns]
        if not top_feats:
            raise ValueError("No numeric features among top_k for correlation heatmap.")
        num = X[top_feats].select_dtypes(include=[np.number])
        corr = num.corr()
        fig = px.imshow(corr, x=corr.columns, y=corr.index, color_continuous_scale="RdBu", zmin=-1, zmax=1)
        fig.update_layout(title=f"Correlation among Top-{top_k} numeric features")
        return fig



from __future__ import annotations
from typing import Dict, Tuple, List, Optional, Iterable
import numpy as np
import pandas as pd

from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance

import plotly.express as px
import plotly.graph_objects as go


# ---------------------------------------------------------------------
# 로깅(권장 포맷) — 외부에서 sink만 연결하면 됨. 여기서는 중복출력 방지용 no-op.
# ---------------------------------------------------------------------
logger.remove()
logger.add(
    sink=lambda msg: None,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} {level} [{name}:{line}] {message}",
)


# =====================================================================
# 1) 중요도 계산 (전역 중요도) — "어떤 X가 F를 지배하는가?"
# =====================================================================
def compute_feature_importance(
    X: pd.DataFrame,
    F: pd.DataFrame,
    *,
    n_estimators: int = 400,
    n_repeats: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    """X(조작조건)→F(목적) 중요도를 안전·간결하게 계산.

    이론/설명(요약):
        - 전역(global) 중요도: "누가 중요하냐?"를 보려면 모델 불가지론적이고 견고한
          permutation importance(PI)가 적합.
        - 트리 기반(RF)은 스케일 불변·비선형에 강함 → 불필요한 스케일링 제거.
        - 카테고리는 One-Hot으로 펼친 뒤, PI는 파생수준에서 산출 → 원 컬럼 단위로 합산해
          사람 해석 가능 형태로 복원(핵심 원리).

    전처리(필수만):
        - 인덱스 정합(교집합/정렬), 목적치 NaN 행 제거
        - 숫자=median, 범주=most_frequent 대체
        - 범주 One-Hot(handle_unknown='ignore')로 안정성 확보

    Args:
        X: (n_samples, n_features) 조작조건 DataFrame.
        F: (n_samples, n_targets) 목적함수 DataFrame.
        n_estimators: RF 트리 수(안정성↑).
        n_repeats: PI 반복 횟수(노이즈 평균).
        random_state: 시드.
        n_jobs: 병렬 처리 코어 수(-1=all).

    Returns:
        importance_df: (원본특성 × 목적) 중요도 표(+ avg_importance). 목적 간 비교 위해 각 목적별로 합=1 정규화.
        models: 목적명 → (전처리+RF) 파이프라인. PDP 등 후속 해석 재사용.
    """
    log = logger.bind(name="feat-imp")

    # --- 인덱스 정합 & 목적 NaN 제거(학습 불가 행 제거) ---
    X = X.copy().sort_index()
    F = F.copy().sort_index()
    idx = X.index.intersection(F.index)
    X, F = X.loc[idx], F.loc[idx]
    ok = ~F.isna().any(axis=1)
    if not ok.all():
        log.info(f"Drop {(~ok).sum()} rows due to NaN in F.")
        X, F = X.loc[ok], F.loc[ok]

    # --- 타입 분리: 숫자/범주 (object/category → 범주로 처리) ---
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.columns.difference(num_cols).tolist()

    # --- 필수 전처리 파이프라인(스케일링 불필요: 트리 기반) ---
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ]) if len(cat_cols) else None

    transformers = []
    if len(num_cols): transformers.append(("num", num_pipe, num_cols))
    if cat_pipe is not None: transformers.append(("cat", cat_pipe, cat_cols))
    preproc = ColumnTransformer(transformers=transformers, remainder="drop", n_jobs=n_jobs)

    # --- 모델 본체: RF (불필요한 가정 최소·안정적 중요도) ---
    def make_pipe() -> Pipeline:
        rf = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs, oob_score=False
        )
        return Pipeline([("pre", preproc), ("rf", rf)])

    # --- 변환 후 특성명 → 원본 컬럼명 매핑(원-핫 파생을 원 컬럼으로 집계) ---
    def get_feature_mapping(pre: ColumnTransformer) -> pd.Series:
        names_after, origin_map = [], []
        for name, trans, cols in pre.transformers_:
            if name == "num":
                for c in cols:
                    names_after.append(c); origin_map.append(c)  # 숫자 특성은 1:1
            elif name == "cat":
                oh: OneHotEncoder = trans.named_steps["oh"]
                oh_names = oh.get_feature_names_out(cols).tolist()  # e.g., col_A, col_B, ...
                names_after.extend(oh_names)
                origin_map.extend([s.split("_", 1)[0] for s in oh_names])  # 원 컬럼명으로 복원
        return pd.Series(origin_map, index=names_after, name="origin")

    models: Dict[str, Pipeline] = {}
    frames: List[pd.DataFrame] = []

    # --- 목적별로 별도 모델 학습 → 목적별 중요도 산출(합=1 정규화) ---
    for tgt in F.columns:
        y = F[tgt].to_numpy()
        pipe = make_pipe()
        pipe.fit(X, y)
        models[tgt] = pipe

        pre_fitted: ColumnTransformer = pipe.named_steps["pre"]
        origin = get_feature_mapping(pre_fitted)

        # 비모수적 전역 중요도: permutation importance
        pi = permutation_importance(
            estimator=pipe, X=X, y=y, n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs
        )
        imp_mean = pd.Series(pi.importances_mean, index=origin.index)
        imp_by_origin = imp_mean.groupby(origin).sum()
        total = imp_by_origin.sum()
        s = imp_by_origin / total if total > 0 else imp_by_origin  # 목적별 합=1 정규화
        frames.append(s.rename(tgt))

    importance_df = pd.concat(frames, axis=1).fillna(0.0)
    importance_df["avg_importance"] = importance_df.mean(axis=1)
    importance_df = importance_df.sort_values("avg_importance", ascending=False)

    return importance_df, models


# =====================================================================
# 2) 시각화 (Heatmap/Bar) — "누가 중요하냐?"를 한눈에
# =====================================================================
def viz_importance_heatmap(
    importance_df: pd.DataFrame,
    top_k: int = 25
) -> go.Figure:
    """상위 top_k 원본 특성 × 목적 중요도 히트맵."""
    cols_targets = [c for c in importance_df.columns if c != "avg_importance"]
    df_ = importance_df.head(top_k).copy()
    fig = px.imshow(
        df_[cols_targets],
        labels=dict(x="Targets", y="Features", color="Importance"),
        x=cols_targets,
        y=df_.index,
        aspect="auto",
        color_continuous_scale="Viridis",
        origin="upper",
    )
    fig.update_layout(title=f"Feature Importance Heatmap (Top {top_k})")
    return fig


def viz_importance_bar(
    importance_df: pd.DataFrame,
    top_k: int = 20
) -> go.Figure:
    """평균 중요도 막대(전략변수 우선순위)."""
    s = importance_df["avg_importance"].head(top_k)
    fig = go.Figure(go.Bar(x=s.index.tolist(), y=s.values.tolist()))
    fig.update_layout(
        title=f"Average Importance (Top {top_k})",
        xaxis_title="Features",
        yaxis_title="Avg Importance",
        xaxis_tickangle=-30
    )
    return fig


# =====================================================================
# 3) PDP — "어떻게 작용하냐?"(방향/포화/비선형/상충 구간)
# =====================================================================
def pdp_plot(
    models: Dict[str, Pipeline],
    X_raw: pd.DataFrame,
    feature: str,
    targets: Optional[Iterable[str]] = None,
    grid_size: int = 60,
    ice_samples: int = 0,
    random_state: int = 42
) -> go.Figure:
    """단일 특성 PDP(+선택 ICE).

    이론/설명(요약):
        - PDP는 "특정 X_i만 바꿔가며" 전 샘플 평균 예측을 관측 → 방향성/포화/비선형 파악.
        - ICE는 개별 샘플 궤적을 덧씌워 이질성(상호작용/조건부 효과) 확인.
        - RF+전처리 파이프라인을 그대로 사용해 스케일·결측·범주 안전성 보장.

    Args:
        models: compute_feature_importance()의 학습 파이프라인 출력.
        X_raw: 원본 X DataFrame(전처리는 파이프라인 내부에서 수행).
        feature: 분석 대상 원본 특성명.
        targets: 그릴 목적 리스트(None이면 전부).
        grid_size: 수치형 PDP 그리드 수.
        ice_samples: ICE 라인 개수(0이면 비활성).
        random_state: ICE 샘플링 시드.

    Returns:
        Plotly Figure.
    """
    assert feature in X_raw.columns, f"feature '{feature}' not found in X"
    tgt_list = list(models.keys()) if targets is None else [t for t in targets if t in models]
    is_numeric = pd.api.types.is_numeric_dtype(X_raw[feature])
    rng = np.random.default_rng(random_state)

    if is_numeric:
        vmin, vmax = X_raw[feature].min(), X_raw[feature].max()
        if not np.isfinite([vmin, vmax]).all():
            raise ValueError(f"Feature '{feature}' has non-finite range.")
        grid = np.linspace(vmin, vmax, grid_size)

        fig = go.Figure()
        for tgt in tgt_list:
            m = models[tgt]
            # grid 값마다 전체 샘플을 해당 값으로 바꿔 예측 → 샘플 평균 → PDP 곡선
            preds_per_grid = []
            for val in grid:
                X_rep = X_raw.copy()
                X_rep[feature] = val
                preds_per_grid.append(m.predict(X_rep))            # shape: (n_samples,)
            preds_mat = np.column_stack(preds_per_grid)            # (n_samples, grid_size)
            y_pdp_line = preds_mat.mean(axis=0)                    # (grid_size,)
            fig.add_trace(go.Scatter(x=grid, y=y_pdp_line, mode="lines", name=f"PDP - {tgt}"))

            # ICE(선택): 개별 샘플 시나리오 궤적
            if ice_samples and ice_samples > 0:
                idx = rng.choice(X_raw.index, size=min(ice_samples, len(X_raw)), replace=False)
                for ridx in idx:
                    xr = X_raw.loc[[ridx]].copy()
                    vals = []
                    for val in grid:
                        xr[feature] = val
                        vals.append(float(m.predict(xr)))
                    fig.add_trace(go.Scatter(
                        x=grid, y=vals, mode="lines",
                        line=dict(width=1, dash="dot"),
                        showlegend=False, name=f"ICE-{tgt}"
                    ))

        fig.update_layout(title=f"PDP for '{feature}' (numeric)", xaxis_title=feature, yaxis_title="Prediction")
        return fig

    # 범주형: 상위 12개 카테고리 평균 예측(막대) — 조작 레벨별 기대 성능 비교
    top_cats = X_raw[feature].astype(str).value_counts().head(12).index.tolist()
    fig = go.Figure()
    for tgt in tgt_list:
        m = models[tgt]
        means = []
        for cat in top_cats:
            X_rep = X_raw.copy()
            X_rep[feature] = cat
            means.append(float(m.predict(X_rep).mean()))
        fig.add_trace(go.Bar(x=top_cats, y=means, name=tgt))
    fig.update_layout(
        barmode="group",
        title=f"PDP for '{feature}' (categorical)",
        xaxis_title=feature, yaxis_title="Prediction", xaxis_tickangle=-30
    )
    return fig


# =====================================================================
# 4) 통합 실행 헬퍼 — 기존 결과 유지 + 시각화까지 한 번에
# =====================================================================
def explain_and_visualize(
    X: pd.DataFrame,
    F: pd.DataFrame,
    *,
    top_k_heatmap: int = 25,
    top_k_bar: int = 20,
    pdp_feature: Optional[str] = None,
    n_estimators: int = 400,
    n_repeats: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline], Dict[str, go.Figure]]:
    """중요도 계산(전역) + 핵심 시각화(히트맵/바 + 선택적 PDP)를 한 번에 수행.

    분석 해석 연결(요약):
        - Heatmap/Bar: "누가 중요?" → 전략변수 우선순위.
        - PDP: "어떻게 작용?" → 방향성/포화/비선형, trade-off 구간.
        - 두 결과를 결합해 "어떤 X가 F를 지배하며, 어떤 레벨에서 상충이 발생하는지"를
          명확히 설명 가능(운전 가이드 도출의 근거).

    Returns:
        importance_df: 원본 특성 × 목적 중요도 표(+ avg_importance).
        models: 목적별 파이프라인.
        figs: {"heatmap", "bar", ("pdp" 선택)} Plotly Figure 사전.
    """
    importance_df, models = compute_feature_importance(
        X, F,
        n_estimators=n_estimators, n_repeats=n_repeats,
        random_state=random_state, n_jobs=n_jobs
    )
    figs: Dict[str, go.Figure] = {
        "heatmap": viz_importance_heatmap(importance_df, top_k=top_k_heatmap),
        "bar": viz_importance_bar(importance_df, top_k=top_k_bar),
    }
    if pdp_feature is not None and pdp_feature in X.columns:
        figs["pdp"] = pdp_plot(models, X, feature=pdp_feature, grid_size=60, ice_samples=0)

    return importance_df, models, figs
