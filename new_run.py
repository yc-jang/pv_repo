from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# 외부 의존: catboost, shap, optuna, scikit-learn (train_test_split, metrics)


class CatBoostRunner:
    """CatBoost 회귀 학습/튜닝/해석 러너 (v6).

    주요 기능:
        - early stopping + 검증기준 과적합 신호 통일(gap, gap_pct, drift)
        - Overfit 리포트(비율 % 표기)
        - Optuna 목적함수에 선택적 과적합 패널티
        - 과적합 신호 기준 재시도(fit_with_guard_retry)
        - SHAP 요약바/워터폴(전역 중요도 순서 고정)

    Notes:
        - 예외는 꼭 필요한 경우에만 raise
        - 로그/프린트는 포함하지 않음(요청 정책)
    """

    # ------------------------------
    # 기본 초기화
    # ------------------------------
    def __init__(
        self,
        use_optuna: bool = True,
        n_trials: int = 50,
        eval_metric: str = "rmse",
        model_name: str = "CatBoostRegressor",
    ) -> None:
        self.use_optuna: bool = use_optuna
        self.n_trials: int = n_trials
        self.eval_metric: str = eval_metric.lower().strip()
        self.model_name: str = model_name

        # 모델/해석기
        self.model: Any = None
        self.explainer: Any = None

        # 데이터 보관
        self.X_train: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_test: Optional[pd.Series] = None
        self.columns: Optional[List[str]] = None

        # SHAP 캐시
        self._shap_values_loaded: Optional[np.ndarray] = None
        self._shap_base_value_loaded: Optional[float] = None
        self._y_pred_test_loaded: Optional[np.ndarray] = None

        # 학습 상태
        self.trained: bool = False
        self.ntree_end_: Optional[int] = None

        # Optuna 결과
        self.best_params: Optional[Dict[str, Any]] = None
        self.optuna_log: Optional[pd.DataFrame] = None

        # 과적합 신호
        self.generalization_gap_: Optional[float] = None
        self.generalization_gap_pct_: Optional[float] = None
        self.generalization_gap_pct_valid_: bool = True
        self.generalization_gap_pct_note_: Optional[str] = None
        self.valid_drift_from_best_: Optional[float] = None
        self.learning_curve_: Optional[Dict[str, List[float]]] = None

        # 가드 요약
        self._guard_summary_: Optional[Dict[str, Any]] = None

    # ------------------------------
    # 데이터 주입
    # ------------------------------
    def _update_data(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> "CatBoostRunner":
        """학습/테스트 데이터를 주입한다."""
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.columns = list(X_train.columns)
        return self

    # ------------------------------
    # CatBoost 기본 파라미터
    # ------------------------------
    def get_default_params(self) -> Dict[str, Any]:
        """CatBoost 기본 파라미터.

        eval_metric:
            - 'rmse' → "RMSE" (손실형)
            - 그 외   → "R2"   (점수형)
        """
        metric_name = "RMSE" if self.eval_metric == "rmse" else "R2"
        return dict(
            loss_function="RMSE",        # 회귀 학습 안정성 위해 손실은 RMSE 고정, 평가는 eval_metric으로
            eval_metric=metric_name,     # 리포팅/early stopping 기준
            use_best_model=True,         # best_iteration 사용
            random_seed=42,
        )

    # ------------------------------
    # 학습 (단순)
    # ------------------------------
    def fit(self, iterations: int = 3000, early_stopping_rounds: int = 200) -> "CatBoostRunner":
        """기본 학습(early stopping 포함)."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("학습 데이터를 먼저 _update_data로 설정하세요.")

        import catboost as cb
        from sklearn.model_selection import train_test_split

        X = self.X_train
        y = self.y_train
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
        train_pool = cb.Pool(X_tr, y_tr)
        valid_pool = cb.Pool(X_val, y_val)

        params = {
            **self.get_default_params(),
            "iterations": iterations,
            "od_type": "Iter",
            "od_wait": early_stopping_rounds,
        }
        self.model = cb.CatBoostRegressor(**params)
        self.model.fit(train_pool, eval_set=valid_pool, verbose=False)

        try:
            self.ntree_end_ = int(self.model.get_best_iteration() + 1)
        except Exception:
            self.ntree_end_ = None

        # explainer 준비(지연 계산)
        self.explainer = None
        self.trained = True
        return self

    # ------------------------------
    # 예측/평가
    # ------------------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """예측."""
        if self.model is None:
            raise ValueError("Model not fitted.")
        ntree = self.ntree_end_ if self.ntree_end_ is not None else None
        return np.asarray(self.model.predict(X, ntree_end=ntree))

    def evaluate(self) -> Dict[str, Any]:
        """테스트셋 평가 결과 반환."""
        if self.X_test is None or self.y_test is None:
            raise ValueError("X_test/y_test가 필요합니다.")
        y_pred = self.predict(self.X_test)
        from sklearn.metrics import r2_score, mean_squared_error

        out = {
            "r2": float(r2_score(self.y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(self.y_test, y_pred))),
            "y_pred": y_pred,
        }
        # 캐시(워터폴 base 추정 등에 사용)
        self._y_pred_test_loaded = y_pred
        return out

    # ------------------------------
    # 과적합 신호 캡처 (v6 통일: 검증 기준 비율)
    # ------------------------------
    def _capture_overfit_signals_from_tmp(self, tmp_model: "cb.CatBoostRegressor") -> None:
        """임시 학습 모델의 결과로 과적합 신호를 수집한다.

        통일 원칙:
            - 비율 격차(gap_pct)는 '검증 점수(Valid) 기준'으로 계산
            - RMSE: gap_pct = (valid - train) / valid
            - R2  : gap_pct = (train - valid) / valid, 단 valid<=0이면 NaN 처리 및 플래그 기록
        """
        import numpy as np

        metric_name = "RMSE" if self.eval_metric == "rmse" else "R2"

        best_it = tmp_model.get_best_iteration()
        best_scores = tmp_model.get_best_score()
        train_best: float = float(best_scores["learn"][metric_name])
        valid_best: float = float(best_scores["validation"][metric_name])

        # 절대 격차(양수 클수록 과적합 신호↑)
        gap_abs = (valid_best - train_best) if metric_name == "RMSE" else (train_best - valid_best)
        self.generalization_gap_ = float(gap_abs)

        # 비율 격차(검증 기준)
        eps = 1e-12
        if metric_name == "RMSE":
            denom = max(valid_best, eps)
            gap_pct = (valid_best - train_best) / denom
            gap_pct_valid = True
            gap_pct_note = None
        else:
            if valid_best <= 0.0:
                gap_pct = float("nan")
                gap_pct_valid = False
                gap_pct_note = "R2 valid<=0 → gap_pct 정의 불가(검증 기준)"
            else:
                denom = max(valid_best, eps)
                gap_pct = (train_best - valid_best) / denom
                gap_pct_valid = True
                gap_pct_note = None

        self.generalization_gap_pct_ = float(gap_pct)
        self.generalization_gap_pct_valid_ = bool(gap_pct_valid)
        self.generalization_gap_pct_note_ = gap_pct_note

        # 러닝커브 & 드리프트
        evals = tmp_model.get_evals_result()
        learn_curve = list(map(float, evals["learn"][metric_name]))
        valid_curve = list(map(float, evals["validation"][metric_name]))
        self.learning_curve_ = {"train": learn_curve, "valid": valid_curve}

        drift = (valid_curve[-1] - valid_curve[best_it]) if metric_name == "RMSE" else (valid_curve[best_it] - valid_curve[-1])
        self.valid_drift_from_best_ = float(drift)

        # 예측 시 사용할 트리 길이
        self.ntree_end_ = int(best_it + 1)

    # ------------------------------
    # Overfit 리포트 (비율 % 표기)
    # ------------------------------
    def overfit_report_df(self) -> pd.DataFrame:
        """과적합 리포트를 DataFrame으로 반환(비율 격차는 % 포맷)."""
        import numpy as np

        gap_pct = getattr(self, "generalization_gap_pct_", np.nan)
        valid_flag = getattr(self, "generalization_gap_pct_valid_", True)
        note = getattr(self, "generalization_gap_pct_note_", None)
        drift = getattr(self, "valid_drift_from_best_", np.nan)
        gap_abs = getattr(self, "generalization_gap_", np.nan)

        df = pd.DataFrame([{
            "일반화 격차(절대)": float(gap_abs) if gap_abs is not None else np.nan,
            "비율 격차(%)": (float(gap_pct) * 100.0) if np.isfinite(gap_pct) else np.nan,
            "비율 격차 유효": bool(valid_flag),
            "비고": note,
            "검증 드리프트": float(drift) if drift is not None else np.nan,
        }])
        return df

    # ------------------------------
    # Optuna 목적함수 (패널티 옵션 포함)
    # ------------------------------
    def objective(
        self,
        trial: "optuna.Trial",
        X_train: pd.DataFrame,
        X_valid: pd.DataFrame,
        y_train: pd.Series,
        y_valid: pd.Series,
        use_gap_pct_penalty: bool = True,
        lambda_gap_pct: float = 0.10,
        use_drift_penalty: bool = False,
        lambda_drift: float = 0.00,
    ) -> float:
        """Optuna 목적함수: 검증 성능 + (선택) 과적합 패널티.

        파라미터 공간(영향도 순 정렬):
            1) learning_rate  : ⬇️(과적합↓, 느림) ↔ iterations ⬆️(충분히 크게)
            2) iterations     : 트리 개수(표현력/시간)
            3) depth          : 트리 깊이(복잡도↑, 과적합↑)
            4) l2_leaf_reg    : 리프 L2 정규화(↑ 과적합↓)
            5) min_data_in_leaf: 리프 최소 표본(↑ 과적합↓, 편향↑ 가능)
            6) random_strength: 분할 점수 랜덤성(↑ 탐색/일반화↑)
            7) bagging_temperature: 샘플 가중치 분산(↑ 다양성↑, 과적합↓)
            8) rsm(colsample_bylevel): 피처 서브샘플 비율(⬇️ 과적합↓)
            9) leaf_estimation_iterations: 리프 추정 반복(↑ 정확도/시간↑)

        Returns:
            penalized objective (RMSE: minimize / R2: minimize(-R2)+penalties)
        """
        import catboost as cb
        import optuna
        from sklearn.metrics import mean_squared_error, r2_score

        metric_name = "RMSE" if self.eval_metric == "rmse" else "R2"

        params: Dict[str, Any] = {
            # 1) shrinkage ↔ capacity
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            # 2) capacity/time
            "iterations": trial.suggest_int("iterations", 800, 5000, step=100),
            # 3) model complexity
            "depth": trial.suggest_int("depth", 6, 10),
            # 4) L2 regularization
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 2.0, 12.0),
            # 5) min samples at leaf
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
            # 6) randomness in splits
            "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
            # 7) Bayesian bootstrap strength
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
            # 8) feature subsample
            "rsm": trial.suggest_float("rsm", 0.6, 1.0),
            # 9) leaf estimation iters
            "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 1, 10),
            # fixed/common
            "eval_metric": metric_name,
            "loss_function": "RMSE",
            "od_type": "Iter",
            "od_wait": 200,
            "use_best_model": True,
            "random_seed": 42,
        }

        train_pool = cb.Pool(X_train, y_train)
        valid_pool = cb.Pool(X_valid, y_valid)

        model = cb.CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=valid_pool, verbose=False)

        best_it = model.get_best_iteration()
        pred = model.predict(valid_pool, ntree_end=best_it)

        if metric_name == "RMSE":
            from math import sqrt
            from sklearn.metrics import mean_squared_error
            main = float(sqrt(mean_squared_error(y_valid, pred)))
            obj = main  # minimize
        else:
            from sklearn.metrics import r2_score
            main = float(r2_score(y_valid, pred))
            obj = -main  # maximize → minimize(-R2)

        # 과적합 패널티
        self._capture_overfit_signals_from_tmp(model)
        gap_pct = self.generalization_gap_pct_
        drift = self.valid_drift_from_best_

        if gap_pct is None or not np.isfinite(gap_pct):
            gap_pct = 1.0  # R2 valid<=0 케이스: 강한 패널티(정책 조정 가능)

        if use_gap_pct_penalty:
            obj += float(lambda_gap_pct) * float(gap_pct)
        if use_drift_penalty:
            obj += float(lambda_drift) * float(drift)

        # 기록
        trial.set_user_attr("valid_main", float(main))
        trial.set_user_attr("gap_pct", float(gap_pct))
        trial.set_user_attr("drift", float(drift))
        trial.set_user_attr("penalized_objective", float(obj))

        return float(obj)

    # ------------------------------
    # Optuna 실행 도우미
    # ------------------------------
    def run_optuna_search(
        self,
        X_train: pd.DataFrame,
        X_valid: pd.DataFrame,
        y_train: pd.Series,
        y_valid: pd.Series,
        n_trials: int = 50,
        use_gap_pct_penalty: bool = True,
        lambda_gap_pct: float = 0.10,
        use_drift_penalty: bool = False,
        lambda_drift: float = 0.00,
    ) -> None:
        """Optuna 탐색 실행(패널티 옵션 포함). 결과는 self.best_params/optuna_log에 저장."""
        import optuna

        direction = "minimize"  # RMSE: minimize / R2: -R2로 minimize 동일 처리
        study = optuna.create_study(direction=direction)
        objective_fn = lambda tr: self.objective(
            tr, X_train, X_valid, y_train, y_valid,
            use_gap_pct_penalty=use_gap_pct_penalty,
            lambda_gap_pct=lambda_gap_pct,
            use_drift_penalty=use_drift_penalty,
            lambda_drift=lambda_drift,
        )
        study.optimize(objective_fn, n_trials=n_trials)

        self.best_params = study.best_params
        self.optuna_log = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))

    # ------------------------------
    # 가드 재시도 학습 (정규화 강화 스케줄)
    # ------------------------------
    def fit_with_guard_retry(
        self,
        test_size: float = 0.1,
        random_state: int = 42,
        base_params: Optional[Dict[str, Any]] = None,
        iterations: int = 3000,
        early_stopping_rounds: int = 200,
        eval_metric: Optional[str] = None,
        gap_pct_threshold: float = 0.15,
        drift_threshold: float = 0.0,
        max_retries: int = 2,
    ) -> "CatBoostRunner":
        """내장 early stopping으로 학습 후, 과적합 신호 기준 초과 시 정규화 강화하여 재학습."""
        import catboost as cb
        from sklearn.model_selection import train_test_split

        if self.X_train is None or self.y_train is None:
            raise ValueError("학습 데이터를 먼저 _update_data로 설정하세요.")

        metric_name = "RMSE" if (eval_metric or self.eval_metric) == "rmse" else "R2"
        is_loss = (metric_name == "RMSE")

        X, y = self.X_train, self.y_train
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
        train_pool = cb.Pool(X_tr, y_tr)
        valid_pool = cb.Pool(X_val, y_val)

        # 정규화 강화 스케줄
        reg_schedule: List[Dict[str, Any]] = [
            {"l2_leaf_reg": 3.0,  "depth": 8, "bagging_temperature": 1.0, "rsm": 1.0, "min_data_in_leaf": 1},
            {"l2_leaf_reg": 6.0,  "depth": 7, "bagging_temperature": 3.0, "rsm": 0.9, "min_data_in_leaf": 10},
            {"l2_leaf_reg": 10.0, "depth": 6, "bagging_temperature": 5.0, "rsm": 0.8, "min_data_in_leaf": 20},
        ][: (max_retries + 1)]

        def _is_better(a: float, b: Optional[float]) -> bool:
            if b is None:
                return True
            return (a < b) if is_loss else (a > b)

        best_model = None
        best_valid = None
        best_attempt = -1

        for attempt, reg in enumerate(reg_schedule):
            params = {
                **self.get_default_params(),
                **(base_params or {}),
                **reg,
                "iterations": iterations,
                "eval_metric": metric_name,
                "use_best_model": True,
                "od_type": "Iter",
                "od_wait": early_stopping_rounds,
                "random_seed": 42,
            }
            model = cb.CatBoostRegressor(**params)
            model.fit(train_pool, eval_set=valid_pool, verbose=False)

            self._capture_overfit_signals_from_tmp(model)
            cur_valid = float(model.get_best_score()["validation"][metric_name])

            if _is_better(cur_valid, best_valid):
                best_model, best_valid, best_attempt = model, cur_valid, attempt

            gap_pct = self.generalization_gap_pct_
            drift = self.valid_drift_from_best_
            gap_ok = (gap_pct is not None and np.isfinite(gap_pct) and (gap_pct < gap_pct_threshold))
            drift_ok = (drift_threshold is None) or (drift is not None and drift <= drift_threshold)

            if gap_ok and drift_ok:
                best_model, best_valid, best_attempt = model, cur_valid, attempt
                break

        self.model = best_model or model
        self.trained = True
        self.columns = X.columns

        try:
            self.ntree_end_ = int(self.model.get_best_iteration() + 1)
        except Exception:
            self.ntree_end_ = None

        # explainer는 지연 준비
        self.explainer = None

        self._guard_summary_ = {
            "attempt": int(best_attempt),
            "best_valid": float(best_valid) if best_valid is not None else None,
            "gap_pct": float(self.generalization_gap_pct_) if self.generalization_gap_pct_ is not None else None,
            "drift": float(self.valid_drift_from_best_) if self.valid_drift_from_best_ is not None else None,
        }
        return self

    # ------------------------------
    # SHAP: 요약바 (Top-k) + Plotly bar
    # ------------------------------
    def shap_summary_bar(self, topk: int = 20) -> Tuple[pd.DataFrame, Optional[go.Figure]]:
        """SHAP 중요도 요약바(Top-k): (df, plotly fig)를 반환."""
        import shap

        X = self.X_test if self.X_test is not None else self.X_train
        if X is None or len(X) == 0:
            raise ValueError("SHAP 계산에 사용할 데이터(X_test 또는 X_train)가 필요합니다.")

        # SHAP 값 캐시/지연 계산
        if self._shap_values_loaded is None:
            if self.explainer is None:
                self.explainer = shap.Explainer(self.model)
            exp = self.explainer(X)
            self._shap_values_loaded = np.asarray(exp.values)
            base_raw = exp.base_values
            self._shap_base_value_loaded = float(np.mean(base_raw)) if np.ndim(base_raw) > 0 else float(base_raw)

        shap_vals = np.asarray(self._shap_values_loaded)  # (n, d)
        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        cols = list(self.columns) if self.columns is not None else list(X.columns)
        imp_df = pd.DataFrame({"feature": cols, "mean_abs_shap": mean_abs}).sort_values(
            "mean_abs_shap", ascending=False
        )
        imp_top = imp_df.head(topk).reset_index(drop=True)

        # Plotly bar
        fig = go.Figure(go.Bar(
            x=imp_top["mean_abs_shap"][::-1],
            y=imp_top["feature"][::-1],
            orientation="h",
            text=[f"{v:.4g}" for v in imp_top["mean_abs_shap"][::-1]],
            textposition="outside",
        ))
        fig.update_layout(
            title=f"[{self.model_name}] SHAP Importance (Top {topk})",
            xaxis_title="Mean |SHAP value|",
            yaxis_title="Feature",
            template="plotly_white",
            margin=dict(l=160, r=40, t=70, b=40),
            showlegend=False,
        )
        return imp_top, fig

    # ------------------------------
    # SHAP: 워터폴 (전역 중요도 순서 고정, Plotly 수평)
    # ------------------------------
    def plot_waterfall_with_mean(
        self,
        idx: Optional[int] = None,
        model_input: Optional[pd.DataFrame] = None,
        condition_columns: Optional[List[str]] = None,
        topk: int = 20,
        aggregate_others: bool = True,
        numeric_mean_fmt: str = "{:.3f}",
        show: bool = False,
    ) -> Tuple[pd.DataFrame, go.Figure]:
        """Plotly 기반 SHAP 워터폴(전역 Mean|SHAP| 순서 고정)."""
        X = model_input if model_input is not None else (self.X_test if self.X_test is not None else self.X_train)
        if X is None or len(X) == 0:
            raise ValueError("X_test 또는 X_train이 필요합니다.")
        cols = list(self.columns) if self.columns is not None else list(X.columns)

        # 전역 중요도 순서
        imp_df, _ = self.shap_summary_bar(topk=topk)
        global_order = imp_df["feature"].tolist()

        # SHAP 값/베이스 확보
        if self._shap_values_loaded is None:
            # shap_summary_bar에서 계산됨
            _ = self.shap_summary_bar(topk=topk)
        shap_vals_all = np.asarray(self._shap_values_loaded)
        base_value = self._shap_base_value_loaded

        # 대상 인덱스
        if idx is None:
            y_pred_all = np.asarray(self.predict(X))
            idx = int(np.argsort(y_pred_all)[len(y_pred_all) // 2])
        if not (0 <= idx < len(X)):
            raise IndexError("idx가 입력 데이터 범위를 벗어났습니다.")

        phi = shap_vals_all[idx, :]
        x_row = X.iloc[idx, :].values

        # 평균/최빈 라벨
        def _avg_label(series: pd.Series) -> str:
            if pd.api.types.is_numeric_dtype(series):
                return numeric_mean_fmt.format(float(series.mean()))
            mode = series.mode(dropna=True)
            return str(mode.iloc[0]) if len(mode) > 0 else "NA"

        mean_map = {c: _avg_label(X[c]) for c in cols}
        annotated = {c: f"(Avg:{mean_map[c]}) | {c}" for c in cols}

        # 전역 순서 + (옵션) 조건 교집합
        if condition_columns:
            sel = [f for f in global_order if f in set(condition_columns)]
        else:
            sel = list(global_order)

        name_to_idx = {c: i for i, c in enumerate(cols)}
        keep_idx = [name_to_idx[f] for f in sel if f in name_to_idx]

        names = [annotated[cols[i]] for i in keep_idx]
        deltas = [float(phi[i]) for i in keep_idx]
        sample_vals = [x_row[i] for i in keep_idx]
        ds_means = [mean_map[cols[i]] for i in keep_idx]

        other_idx = [i for i in range(len(cols)) if i not in keep_idx]
        if aggregate_others and len(other_idx) > 0:
            other_sum = float(np.sum(phi[other_idx]))
            other_df = X.iloc[:, other_idx]
            if len(other_idx) > 0 and all(pd.api.types.is_numeric_dtype(other_df[c]) for c in other_df.columns):
                other_sample_scalar = float(np.mean(other_df.iloc[idx, :]))
                other_mean_text = numeric_mean_fmt.format(float(other_df.values.mean()))
            else:
                flat_s = pd.Series(other_df.iloc[idx, :]) if len(other_idx) > 0 else pd.Series([])
                mode_all = flat_s.mode(dropna=True)
                other_sample_scalar = str(mode_all.iloc[0]) if len(mode_all) > 0 else "NA"
                other_mean_text = "mixed"
            names.append(f"(Avg:{other_mean_text}) | (Other Features)")
            deltas.append(other_sum)
            sample_vals.append(other_sample_scalar)
            ds_means.append(other_mean_text)

        # 누적 경로(전역 순서 기준)
        base = float(base_value) if base_value is not None else float(np.mean(self._y_pred_test_loaded - shap_vals_all.sum(axis=1)))
        cum_before, cum_after = [], []
        cur = base
        for d in deltas:
            start = cur if d >= 0 else cur + d
            cum_before.append(start)
            cum_after.append(start + abs(d))
            cur = cur + d
        pred = cur

        abs_d = np.abs(deltas)
        ranks = list(range(1, len(deltas) + 1))

        df = pd.DataFrame({
            "feature": names,
            "delta": deltas,
            "abs_delta": abs_d,
            "sample_value": sample_vals,
            "dataset_mean": ds_means,
            "rank": ranks,
            "base": base,
            "prediction": pred,
            "idx": idx,
            "cum_before": cum_before,
            "cum_after": cum_after,
        })

        # Plotly 수평 워터폴
        fig = go.Figure()
        for i, yi in enumerate(names):
            d = deltas[i]
            start = cum_before[i]
            width = abs(d)
            color = "rgba(214,39,40,0.85)" if d < 0 else "rgba(31,119,180,0.85)"
            mu_disp = ds_means[i]
            feat_disp = yi.split(" | ", 1)[1] if " | " in yi else yi
            v_disp = sample_vals[i]
            v_txt = numeric_mean_fmt.format(float(v_disp)) if isinstance(v_disp, (int, float, np.floating)) else str(v_disp)
            hover = f"{feat_disp}<br>μ={mu_disp} | v={v_txt} | Δ={d:+.4g}"

            fig.add_trace(go.Bar(
                y=[yi],
                x=[width],
                base=[start],
                orientation="h",
                marker_color=color,
                hovertemplate=hover + "<extra></extra>",
                text=[f"{d:+.4g}"],
                textposition="outside",
            ))

        fig.add_vline(x=base, line=dict(color="gray", dash="dot"),
                      annotation_text="base", annotation_position="top left")
        fig.add_vline(x=pred, line=dict(color="green", dash="dot"),
                      annotation_text="prediction", annotation_position="top right")

        fig.update_layout(
            title=f"[{self.model_name}] SHAP Waterfall (idx={idx}, global-order top{topk}"
                  f"{', cond' if (condition_columns and len(condition_columns)>0) else ''})",
            xaxis_title="Model output",
            yaxis_title="Feature (Avg | name)",
            barmode="overlay",
            template="plotly_white",
            margin=dict(l=160, r=40, t=70, b=40),
            showlegend=False,
        )
        if show:
            fig.show()
        return df, fig
