from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import catboost as cb
import shap
import optuna

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error


class CatBoostRunner:
    """CatBoost 회귀 러너 클래스.

    v1 기반:
      - 평가지표 정규화(eval_metric_key)
      - best iteration(+1) 일관 예측
      - cat_features 자동/수동 처리
      - Train+Valid 전체 재학습

    (제안 추가) 오버피팅 판단 근거 수집:
      - 일반화 격차(Train vs Valid @ best)
      - 검증 드리프트(최적 이후 검증 지표 악화)
      - 한국어 DataFrame 리포트 제공(overfit_report_df)

    Attributes:
        eval_metric (str): 사용자 입력 평가지표 원문.
        eval_metric_key (str): 정규화된 평가지표 키('rmse' 또는 'r2').
        model (cb.CatBoostRegressor): 최종 학습 모델.
        explainer (shap.Explainer | None): SHAP explainer.
        _shap_value: 내부 SHAP 저장소(선택).
        X_train, X_test, y_train, y_test: 데이터 홀더.
        trained (bool): 학습 완료 여부.
        use_optuna (bool): Optuna 사용 여부.
        n_trials (int): Optuna trial 횟수.
        best_params (dict | None): 탐색된 최적 파라미터.
        optuna_log (pd.DataFrame | None): Optuna 로그.
        columns (pd.Index | None): 학습 피처 컬럼 순서.
        cat_features (List[int|str] | None): 카테고리 피처 지정.
        ntree_end_ (int | None): 최종 예측 시 사용할 트리 개수(best_it+1).

        # 오버피팅 신호 수집 필드
        train_metric_at_best (float | None)
        valid_metric_at_best (float | None)
        generalization_gap_ (float | None)
        generalization_gap_pct_ (float | None)
        valid_drift_from_best_ (float | None)
        learning_curve_ (dict | None)
    """

    # ---- 생성자 ----
    def __init__(
        self,
        use_optuna: bool = True,
        n_trials: int = 30,
        eval_metric: str = 'rmse',
        cat_features: Optional[List[Union[int, str]]] = None
    ):
        """Init.

        Args:
            use_optuna: Optuna 사용 여부.
            n_trials: Optuna trial 횟수.
            eval_metric: 'rmse' 또는 'r2' 등(대소문자/공백 무시).
            cat_features: 카테고리 피처 지정(컬럼명 또는 인덱스). 미지정 시 자동 추론.
        """
        self.eval_metric: str = eval_metric
        self.eval_metric_key: str = self._normalize_eval_metric(eval_metric)
        self.model: cb.CatBoostRegressor = cb.CatBoostRegressor(**self.get_default_params())
        self.explainer: Optional[shap.Explainer] = None
        self._shap_value = None

        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.trained: bool = False

        self.use_optuna: bool = use_optuna
        self.n_trials: int = n_trials
        self.best_params: Optional[Dict[str, Any]] = None
        self.optuna_log: Optional[pd.DataFrame] = None

        self.columns: Optional[pd.Index] = None
        self.cat_features: Optional[List[Union[int, str]]] = cat_features
        self.ntree_end_: Optional[int] = None  # 최종 예측에서 사용할 트리 개수(best_it+1)

        # --- 오버피팅 신호 필드 초기화 ---
        self.train_metric_at_best: Optional[float] = None
        self.valid_metric_at_best: Optional[float] = None
        self.generalization_gap_: Optional[float] = None
        self.generalization_gap_pct_: Optional[float] = None
        self.valid_drift_from_best_: Optional[float] = None
        self.learning_curve_: Optional[Dict[str, List[float]]] = None

    # ---------- 내부 유틸 ----------
    def _normalize_eval_metric(self, metric: str) -> str:
        """평가지표 문자열 정규화.

        Args:
            metric: 사용자 입력 평가지표.

        Returns:
            정규화된 키. 기본은 'rmse', 그 외는 'r2'로 처리.
        """
        key = (metric or '').strip().lower()
        return 'rmse' if key == 'rmse' else 'r2'

    def _metric_name(self) -> str:
        """CatBoost evals_result/get_best_score 키 이름."""
        return 'RMSE' if self.eval_metric_key == 'rmse' else 'R2'

    def _infer_cat_features(self, X: pd.DataFrame) -> List[Union[int, str]]:
        """데이터프레임으로부터 카테고리 피처를 추론한다.

        object/category dtype을 카테고리로 간주.
        """
        # dtype 기반 추론
        cats = [c for c in X.columns if str(X[c].dtype) in ('object', 'category')]
        return cats

    def _get_eval_score(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """메인 평가지표 점수 계산.

        Args:
            y_true: 정답 타깃.
            y_pred: 예측값.

        Returns:
            점수( rmse → 낮을수록 좋음 / r2 → 높을수록 좋음 ).
        """
        # 핵심 지표 분기
        return (root_mean_squared_error(y_true, y_pred)
                if self.eval_metric_key == 'rmse' else r2_score(y_true, y_pred))

    def _resolve_cat_features(self, X: pd.DataFrame) -> List[Union[int, str]]:
        """cat_features 최종 결정."""
        if self.cat_features is not None:
            return self.cat_features
        return self._infer_cat_features(X)

    def get_default_params(self) -> Dict[str, Any]:
        """CatBoost 기본 파라미터."""
        return dict(
            loss_function='RMSE',
            eval_metric="RMSE" if self.eval_metric_key == 'rmse' else 'R2',
            use_best_model=True,
            random_seed=42
        )

    # ---------- 데이터 주입 ----------
    def _update_data(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> "CatBoostRunner":
        """학습/평가 데이터 설정."""
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        return self

    # ---------- Optuna 목적함수 ----------
    def objective(
        self,
        trial: optuna.Trial,
        X_train: pd.DataFrame, X_valid: pd.DataFrame,
        y_train: pd.Series, y_valid: pd.Series
    ) -> float:
        """Optuna 목적함수: trial별 임시 모델 학습/평가."""
        params: Dict[str, Any] = {
            **self.get_default_params(),
            # 하이퍼파라미터 탐색공간은 이후 승인 시 추가
        }

        cats = self._resolve_cat_features(X_train)
        train_pool = cb.Pool(X_train, y_train, cat_features=cats if len(cats) > 0 else None)
        valid_pool = cb.Pool(X_valid, y_valid, cat_features=cats if len(cats) > 0 else None)

        model = cb.CatBoostRegressor(**params)
        model.fit(
            train_pool,
            eval_set=valid_pool,
            early_stopping_rounds=100,
            verbose=False
        )

        best_it = model.get_best_iteration()
        ntree_end = (best_it + 1) if best_it is not None else None
        preds = model.predict(valid_pool, ntree_end=ntree_end)

        # 주 평가점수
        main_score = self._get_eval_score(y_valid, preds)

        # 부가지표를 trial attrs로 저장(옵션)
        valid_rmse = float(root_mean_squared_error(y_valid, preds))
        valid_r2 = float(r2_score(y_valid, preds))
        trial.set_user_attr("valid_rmse", valid_rmse)
        trial.set_user_attr("valid_r2", valid_r2)
        trial.set_user_attr("best_iteration", float(best_it if best_it is not None else -1))

        return main_score

    # ---------- 오버피팅 신호 수집 ----------
    def _capture_overfit_signals_from_tmp(self, tmp_model: cb.CatBoostRegressor) -> None:
        """홀드아웃 학습(tmp_model) 결과로 오버피팅 관련 '핵심' 신호를 수집한다.

        수집 항목:
          - train/valid best 점수
          - 일반화 격차 및 비율
          - 검증 드리프트
          - 러닝커브(train/valid)
        """
        metric_name = self._metric_name()

        # 학습/검증 베스트 점수
        best_scores = tmp_model.get_best_score()  # {'learn': {'RMSE': ...}, 'validation': {'RMSE': ...}}
        train_best = float(best_scores['learn'][metric_name])
        valid_best = float(best_scores['validation'][metric_name])

        self.train_metric_at_best = train_best
        self.valid_metric_at_best = valid_best

        # 일반화 격차 및 정규화 비율
        if self.eval_metric_key == 'rmse':
            gap = valid_best - train_best
            gap_pct = gap / max(valid_best, 1e-12)
        else:  # r2
            gap = train_best - valid_best
            denom = abs(train_best) if abs(train_best) > 1e-12 else 1.0
            gap_pct = gap / denom

        self.generalization_gap_ = float(gap)
        self.generalization_gap_pct_ = float(gap_pct)

        # 검증 드리프트
        evals = tmp_model.get_evals_result()  # {'learn': {'RMSE': [...]}, 'validation': {'RMSE': [...]} }
        learn_curve = evals['learn'][metric_name]
        valid_curve = evals['validation'][metric_name]
        best_it = tmp_model.get_best_iteration()

        drift: Optional[float] = None
        if best_it is not None and len(valid_curve) > 0:
            if self.eval_metric_key == 'rmse':
                drift = float(valid_curve[-1] - valid_curve[best_it])  # 양수↑ 나쁨
            else:
                drift = float(valid_curve[best_it] - valid_curve[-1])  # 양수↑ 나쁨

        self.valid_drift_from_best_ = drift
        self.learning_curve_ = {'train': learn_curve, 'valid': valid_curve}

    # ---------- 학습 ----------
    def fit(self) -> "CatBoostRunner":
        """모델 학습.

        흐름:
          1) Train/Valid split
          2) (옵션) Optuna 탐색으로 best_params 획득
          3) best_params로 홀드아웃(tmp_model) 학습 → best_it, 러닝커브, 오버핏 신호 수집
          4) best_it+1 반복수로 Train+Valid 전체 재학습(최종 모델 고정)
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data is not set. Call _update_data first.")

        X = self.X_train.copy()
        y = self.y_train.copy()
        self.columns = X.columns

        best_params = self.get_default_params()

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.1, random_state=42
        )

        cats = self._resolve_cat_features(X_train)

        # ---- 1) 탐색 & 홀드아웃 학습 ----
        if self.use_optuna:
            direction = 'minimize' if self.eval_metric_key == 'rmse' else 'maximize'
            study = optuna.create_study(direction=direction)

            objective_fn = lambda trial: self.objective(trial, X_train, X_valid, y_train, y_valid)
            study.optimize(objective_fn, n_trials=self.n_trials)

            best_params.update(study.best_params)
            self.optuna_log = study.trials_dataframe(attrs=('number', 'value', 'params', 'user_attrs'))
            self.best_params = study.best_params

        # 홀드아웃에서 best_it 계산 및 신호 수집을 위해 한 번 학습
        train_pool = cb.Pool(X_train, y_train, cat_features=cats if len(cats) > 0 else None)
        valid_pool = cb.Pool(X_valid, y_valid, cat_features=cats if len(cats) > 0 else None)

        tmp_model = cb.CatBoostRegressor(**best_params)
        tmp_model.fit(
            train_pool,
            eval_set=valid_pool,
            early_stopping_rounds=100,
            verbose=False
        )

        best_it = tmp_model.get_best_iteration()
        self.ntree_end_ = (best_it + 1) if best_it is not None else None  # 최종 예측용 저장

        # --- 오버피팅 신호 수집 ---
        self._capture_overfit_signals_from_tmp(tmp_model)

        # ---- 2) 전체(Train+Valid) 재학습 ----
        full_X = pd.concat([X_train, X_valid], axis=0)
        full_y = pd.concat([y_train, y_valid], axis=0)
        full_cats = self._resolve_cat_features(full_X)

        final_params = dict(best_params)
        # 전체 재학습에서는 use_best_model 영향 배제를 위해 False 권장
        final_params['use_best_model'] = False
        if self.ntree_end_ is not None:
            # best_it+1 만큼의 반복수로 고정
            final_params['iterations'] = self.ntree_end_

        full_pool = cb.Pool(full_X, full_y, cat_features=full_cats if len(full_cats) > 0 else None)

        self.model = cb.CatBoostRegressor(**final_params)
        self.model.fit(full_pool, verbose=False)

        self.explainer = shap.Explainer(self.model)
        self.trained = True
        return self

    # ---------- 예측/평가 ----------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """예측.

        컬럼 순서를 학습 시점과 동일하게 정렬하고,
        가능하면 best_it+1 개수로 예측한다.
        """
        if self.model is None:
            raise ValueError("Model not Fitted")

        # 학습 컬럼 순서 강제 정렬로 매핑 오류 방지
        if self.columns is not None:
            X = X.reindex(columns=self.columns, copy=False)

        ntree_end = self.ntree_end_
        return self.model.predict(X, ntree_end=ntree_end)

    def evaluate(self) -> Dict[str, Any]:
        """테스트셋 평가.

        Returns:
            r2, rmse, y_pred 딕셔너리.
        """
        if self.X_test is None or self.y_test is None:
            raise ValueError("Test data is not set. Call _update_data first.")

        X = self.X_test.copy()
        y = self.y_test.copy()
        y_pred = self.predict(X)
        return {
            "r2": r2_score(y, y_pred),
            "rmse": root_mean_squared_error(y, y_pred),
            "y_pred": y_pred
        }

    # ---------- 오버피팅 리포트(한글/DF) ----------
    def overfit_report_df(
        self,
        rmse_gap_warn: float = 0.10,
        r2_gap_warn: float = 0.05
    ) -> pd.DataFrame:
        """오버피팅 신호 요약 리포트를 한국어 컬럼의 DataFrame으로 반환한다.

        Args:
            rmse_gap_warn: RMSE 기준 비율 격차 경고 임계값(기본 10%).
            r2_gap_warn: R2 기준 절대 격차 경고 임계값(기본 0.05).

        Returns:
            한국어 컬럼의 단일 로우 DataFrame.
        """
        if self.train_metric_at_best is None or self.valid_metric_at_best is None:
            raise ValueError("오버피팅 신호가 수집되지 않았습니다. fit()을 먼저 수행하세요.")

        metric = 'rmse' if self.eval_metric_key == 'rmse' else 'r2'

        if metric == 'rmse':
            warn_gap = (self.generalization_gap_pct_ is not None and
                        self.generalization_gap_pct_ > rmse_gap_warn)
            warn_drift = (self.valid_drift_from_best_ is not None and
                          self.valid_drift_from_best_ > 0.0)
        else:
            warn_gap = (self.generalization_gap_ is not None and
                        self.generalization_gap_ > r2_gap_warn)
            warn_drift = (self.valid_drift_from_best_ is not None and
                          self.valid_drift_from_best_ > 0.0)

        row = {
            "평가지표": metric.upper(),  # 'RMSE' 또는 'R2'
            "학습 점수(@best)": self.train_metric_at_best,
            "검증 점수(@best)": self.valid_metric_at_best,
            "일반화 격차": self.generalization_gap_,
            "일반화 격차(비율)": self.generalization_gap_pct_,
            "검증 드리프트(최적 이후)": self.valid_drift_from_best_,
            "반복 수(ntree_end)": self.ntree_end_,
            "경고(격차)": bool(warn_gap),
            "경고(드리프트)": bool(warn_drift),
        }
        return pd.DataFrame([row])

    def overfit_report(self,
                       rmse_gap_warn: float = 0.10,
                       r2_gap_warn: float = 0.05) -> Dict[str, Any]:
        """오버피팅 신호 요약 리포트(딕셔너리). CSV 저장 전 전처리 용이."""
        df = self.overfit_report_df(rmse_gap_warn=rmse_gap_warn, r2_gap_warn=r2_gap_warn)
        return df.iloc[0].to_dict()
