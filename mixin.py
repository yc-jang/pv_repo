from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Literal
import os
import pickle
import joblib
import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import catboost as cb
import shap
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


# =========================
# Utilities
# =========================
def _now_iso() -> str:
    """Return current time in ISO string (seconds)."""
    return dt.datetime.now().isoformat(timespec="seconds")


def _rmse(y_true: Union[pd.Series, np.ndarray],
          y_pred: Union[pd.Series, np.ndarray]) -> float:
    """RMSE with sklearn compatibility across versions."""
    return float(mean_squared_error(y_true, y_pred, squared=False))


# =========================
# ResultProtocolMixin
# =========================
class ResultProtocolMixin:
    """Provide standardized result DataFrames/plots for trained or loaded models.

    Required attributes/methods on subclasses:
        - self.model_name: str
        - self.columns: Optional[pd.Index]
        - self.ntree_end_: Optional[int]
        - self.X_test: Optional[pd.DataFrame]
        - self.y_test: Optional[pd.Series]
        - self.model: Any
        - self.scores() -> Dict[str, float]
        - self.predict(X: pd.DataFrame) -> np.ndarray
    """

    RESULTS_COLUMNS_LONG = [
        "model", "dataset", "index", "y_true", "y_pred", "residual", "r2", "rmse", "timestamp"
    ]

    def predict_vs_actual_df(self) -> pd.DataFrame:
        """Return standardized long-form DataFrame for y_true vs y_pred on test set.

        Returns:
            pd.DataFrame: columns ['model','dataset','index','y_true','y_pred','residual'] (+ r2/rmse/timestamp in get_results)
        """
        if self.X_test is None or self.y_test is None:
            raise ValueError("X_test / y_test must be set before generating results.")

        X = self.X_test.copy()
        y_true = self.y_test.copy()
        y_pred = self.predict(X)  # column order & ntree_end handled by subclass

        df = pd.DataFrame({
            "model": self.model_name,
            "dataset": "test",
            "index": y_true.index,
            "y_true": np.asarray(y_true),
            "y_pred": np.asarray(y_pred),
        })
        df["residual"] = df["y_true"] - df["y_pred"]
        return df

    def get_results(self, wide: bool = False, id_col: Optional[str] = None) -> pd.DataFrame:
        """Return standardized results for easy concat across models; supports wide format.

        Args:
            wide: If True, return wide format with a single y_true and y_pred_<model> column.
            id_col: Optional ID column to include, taken from X_test aligned by index.

        Returns:
            pd.DataFrame: long or wide format results.
        """
        base = self.predict_vs_actual_df()
        try:
            sc = self.scores()  # {'r2':..., 'rmse':...}
        except Exception:
            sc = {}
        base["r2"] = sc.get("r2", np.nan)
        base["rmse"] = sc.get("rmse", np.nan)
        base["timestamp"] = _now_iso()

        if id_col and self.X_test is not None and id_col in self.X_test.columns:
            base[id_col] = self.X_test.loc[base["index"], id_col].values

        if not wide:
            cols = self.RESULTS_COLUMNS_LONG + ([id_col] if id_col else [])
            return base[cols].copy()

        wide_df = base[["index", "y_true", "y_pred"]].copy()
        wide_df = wide_df.rename(columns={"y_pred": f"y_pred_{self.model_name}"})
        wide_df = wide_df.drop_duplicates(subset=["index"])
        if id_col and id_col in base.columns:
            wide_df[id_col] = base.drop_duplicates(subset=["index"])[id_col].values
        return wide_df

    @staticmethod
    def concat_results(results: List[pd.DataFrame], wide: bool = False) -> pd.DataFrame:
        """Concatenate results from multiple models.

        Args:
            results: List of DataFrames from get_results().
            wide: If True, merge wide DFs by index/y_true; otherwise simple row concat.

        Returns:
            pd.DataFrame
        """
        if not results:
            return pd.DataFrame()
        if not wide:
            return pd.concat(results, ignore_index=True)

        # wide merge by index + y_true (robust for aligned test set)
        out = results[0]
        for df in results[1:]:
            join_keys = [k for k in ["index", "y_true"] if k in out.columns and k in df.columns]
            out = out.merge(df, on=join_keys, how="outer")
        return out

    # ---------- Plots ----------
    def plot_predictions_line(self,
                              sort_by_index: bool = True,
                              sample: Optional[int] = None,
                              title: Optional[str] = None) -> go.Figure:
        """Line plot of y_true vs y_pred on test set."""
        df = self.predict_vs_actual_df()
        if sort_by_index:
            df = df.sort_values("index")
        if sample is not None and sample > 0:
            df = df.head(sample)

        x = list(range(len(df)))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=df["y_true"], mode="lines", name="정답(y_true)"))
        fig.add_trace(go.Scatter(x=x, y=df["y_pred"], mode="lines", name="예측(y_pred)"))
        fig.update_layout(
            title=title or f"[{self.model_name}] 예측 vs 정답 (테스트셋)",
            xaxis_title="순서(정렬/샘플 적용)",
            yaxis_title="값",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig

    def plot_residuals_scatter(self, title: Optional[str] = None) -> go.Figure:
        """Scatter plot of residuals vs y_true."""
        df = self.predict_vs_actual_df()
        fig = go.Figure(go.Scatter(x=df["y_true"], y=df["residual"], mode="markers", name="잔차"))
        fig.update_layout(
            title=title or f"[{self.model_name}] 잔차 산점도 (테스트셋)",
            xaxis_title="y_true",
            yaxis_title="residual (y_true - y_pred)",
            template="plotly_white"
        )
        return fig

    def plot_pred_vs_true_scatter(self, title: Optional[str] = None) -> go.Figure:
        """Scatter plot of y_pred vs y_true with y=x baseline."""
        df = self.predict_vs_actual_df()
        lim_min = float(np.nanmin([df["y_true"].min(), df["y_pred"].min()]))
        lim_max = float(np.nanmax([df["y_true"].max(), df["y_pred"].max()]))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["y_true"], y=df["y_pred"], mode="markers", name="예측 vs 정답"))
        fig.add_trace(go.Scatter(x=[lim_min, lim_max], y=[lim_min, lim_max],
                                 mode="lines", name="y = x", line=dict(dash="dash")))
        fig.update_layout(
            title=title or f"[{self.model_name}] 예측 vs 정답 (테스트셋)",
            xaxis_title="y_true",
            yaxis_title="y_pred",
            template="plotly_white"
        )
        return fig


# =========================
# CatBoostRunner (v3)
# =========================
class CatBoostRunner(ResultProtocolMixin):
    """CatBoost regression runner (v3).

    Features:
        - v2 core (metric normalization, best_it+1 consistency, cat_features, full re-train)
        - Save/Load artifact with consistent schema
        - Standard results DF/plots via ResultProtocolMixin
        - Overfitting signals (generalization gap & validation drift) with Korean DataFrame report
        - SHAP importance DF and CatBoost native feature importance DF
    """

    # ---------- Init ----------
    def __init__(self,
                 use_optuna: bool = True,
                 n_trials: int = 30,
                 eval_metric: str = "rmse",
                 cat_features: Optional[List[Union[int, str]]] = None):
        self.model_name: str = "CatBoost"

        # Metric normalization
        self.eval_metric: str = eval_metric
        self.eval_metric_key: str = self._normalize_eval_metric(eval_metric)

        # Model & data holders
        self.model: Optional[cb.CatBoostRegressor] = cb.CatBoostRegressor(**self.get_default_params())
        self.explainer: Optional[shap.Explainer] = None

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
        self.ntree_end_: Optional[int] = None  # best_it + 1 for consistent inference

        # Overfitting signals
        self.train_metric_at_best: Optional[float] = None
        self.valid_metric_at_best: Optional[float] = None
        self.generalization_gap_: Optional[float] = None
        self.generalization_gap_pct_: Optional[float] = None
        self.valid_drift_from_best_: Optional[float] = None
        self.learning_curve_: Optional[Dict[str, List[float]]] = None

    # ---------- Utils / Config ----------
    def _normalize_eval_metric(self, metric: str) -> str:
        """Normalize eval metric string to 'rmse' or 'r2'."""
        key = (metric or "").strip().lower()
        return "rmse" if key == "rmse" else "r2"

    def _metric_name(self) -> str:
        """CatBoost metric name used in evals_result & best_score."""
        return "RMSE" if self.eval_metric_key == "rmse" else "R2"

    def _infer_cat_features(self, X: pd.DataFrame) -> List[Union[int, str]]:
        """Infer categorical features from dtypes (object/category)."""
        return [c for c in X.columns if str(X[c].dtype) in ("object", "category")]

    def _get_eval_score(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Return main metric score."""
        return _rmse(y_true, y_pred) if self.eval_metric_key == "rmse" else float(r2_score(y_true, y_pred))

    def _resolve_cat_features(self, X: pd.DataFrame) -> List[Union[int, str]]:
        """Resolve final cat_features list."""
        return self.cat_features if self.cat_features is not None else self._infer_cat_features(X)

    def get_default_params(self) -> Dict[str, Any]:
        """Default CatBoost parameters."""
        return dict(
            loss_function="RMSE",
            eval_metric=("RMSE" if self.eval_metric_key == "rmse" else "R2"),
            use_best_model=True,
            random_seed=42
        )

    # ---------- Data Injection ----------
    def _update_data(self,
                     X_train: pd.DataFrame,
                     X_test: pd.DataFrame,
                     y_train: pd.Series,
                     y_test: pd.Series) -> "CatBoostRunner":
        """Set train/test datasets."""
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        return self

    # ---------- Optuna Objective ----------
    def objective(self,
                  trial: optuna.Trial,
                  X_train: pd.DataFrame, X_valid: pd.DataFrame,
                  y_train: pd.Series, y_valid: pd.Series) -> float:
        """Optuna objective: train a temporary model and return validation score."""
        params: Dict[str, Any] = {
            **self.get_default_params(),
            # TODO: add search space if needed
        }

        cats = self._resolve_cat_features(X_train)
        train_pool = cb.Pool(X_train, y_train, cat_features=cats if len(cats) > 0 else None)
        valid_pool = cb.Pool(X_valid, y_valid, cat_features=cats if len(cats) > 0 else None)

        model = cb.CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=100, verbose=False)

        best_it = model.get_best_iteration()
        ntree_end = (best_it + 1) if best_it is not None else None
        preds = model.predict(valid_pool, ntree_end=ntree_end)

        score = self._get_eval_score(y_valid, preds)

        trial.set_user_attr("valid_rmse", float(_rmse(y_valid, preds)))
        trial.set_user_attr("valid_r2", float(r2_score(y_valid, preds)))
        trial.set_user_attr("best_iteration", float(best_it if best_it is not None else -1))
        return score

    # ---------- Overfitting Signals ----------
    def _capture_overfit_signals_from_tmp(self, tmp_model: cb.CatBoostRegressor) -> None:
        """Collect generalization gap & validation drift from temporary holdout model."""
        metric_name = self._metric_name()
        best_scores = tmp_model.get_best_score()
        train_best = float(best_scores["learn"][metric_name])
        valid_best = float(best_scores["validation"][metric_name])

        self.train_metric_at_best = train_best
        self.valid_metric_at_best = valid_best

        if self.eval_metric_key == "rmse":
            gap = valid_best - train_best
            gap_pct = gap / max(valid_best, 1e-12)
        else:
            gap = train_best - valid_best
            denom = abs(train_best) if abs(train_best) > 1e-12 else 1.0
            gap_pct = gap / denom

        self.generalization_gap_ = float(gap)
        self.generalization_gap_pct_ = float(gap_pct)

        evals = tmp_model.get_evals_result()
        learn_curve = evals["learn"][metric_name]
        valid_curve = evals["validation"][metric_name]
        best_it = tmp_model.get_best_iteration()

        drift: Optional[float] = None
        if best_it is not None and len(valid_curve) > 0:
            drift = (float(valid_curve[-1] - valid_curve[best_it]) if self.eval_metric_key == "rmse"
                     else float(valid_curve[best_it] - valid_curve[-1]))

        self.valid_drift_from_best_ = drift
        self.learning_curve_ = {"train": learn_curve, "valid": valid_curve}

    # ---------- Fit ----------
    def fit(self) -> "CatBoostRunner":
        """Train model: (optional) Optuna -> holdout tmp -> full re-train with fixed iterations."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data is not set. Call _update_data first.")

        X = self.X_train.copy()
        y = self.y_train.copy()
        self.columns = X.columns

        best_params = self.get_default_params()

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)
        cats = self._resolve_cat_features(X_train)

        if self.use_optuna:
            direction = "minimize" if self.eval_metric_key == "rmse" else "maximize"
            study = optuna.create_study(direction=direction)
            objective_fn = lambda t: self.objective(t, X_train, X_valid, y_train, y_valid)
            study.optimize(objective_fn, n_trials=self.n_trials)
            best_params.update(study.best_params)
            self.optuna_log = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
            self.best_params = study.best_params

        # Temporary holdout model for best_it and curves
        train_pool = cb.Pool(X_train, y_train, cat_features=cats if len(cats) > 0 else None)
        valid_pool = cb.Pool(X_valid, y_valid, cat_features=cats if len(cats) > 0 else None)

        tmp_model = cb.CatBoostRegressor(**best_params)
        tmp_model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=100, verbose=False)

        best_it = tmp_model.get_best_iteration()
        self.ntree_end_ = (best_it + 1) if best_it is not None else None
        self._capture_overfit_signals_from_tmp(tmp_model)

        # Full (Train + Valid) re-train with fixed iterations
        full_X = pd.concat([X_train, X_valid], axis=0)
        full_y = pd.concat([y_train, y_valid], axis=0)
        full_cats = self._resolve_cat_features(full_X)

        final_params = dict(best_params)
        final_params["use_best_model"] = False
        if self.ntree_end_ is not None:
            final_params["iterations"] = self.ntree_end_

        full_pool = cb.Pool(full_X, full_y, cat_features=full_cats if len(full_cats) > 0 else None)
        self.model = cb.CatBoostRegressor(**final_params)
        self.model.fit(full_pool, verbose=False)

        try:
            self.explainer = shap.Explainer(self.model)
        except Exception:
            self.explainer = None

        self.trained = True
        return self

    # ---------- Predict / Evaluate ----------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict with column-order safety and fixed ntree_end if available."""
        if self.model is None:
            raise ValueError("Model not Fitted/Loaded.")
        if self.columns is not None:
            X = X.reindex(columns=self.columns, copy=False)
        return self.model.predict(X, ntree_end=self.ntree_end_)

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate on test set; returns r2, rmse, and y_pred array."""
        if self.X_test is None or self.y_test is None:
            raise ValueError("Test data is not set. Call _update_data first.")
        X = self.X_test.copy()
        y = self.y_test.copy()
        y_pred = self.predict(X)
        return {"r2": float(r2_score(y, y_pred)), "rmse": _rmse(y, y_pred), "y_pred": y_pred}

    def scores(self) -> Dict[str, float]:
        """Return r2 & rmse scores on test set; empty dict if unavailable."""
        try:
            out = self.evaluate()
            return {"r2": float(out.get("r2", np.nan)), "rmse": float(out.get("rmse", np.nan))}
        except Exception:
            return {}

    # ---------- Overfitting Report (Korean DataFrame) ----------
    def overfit_report_df(self,
                          rmse_gap_warn: float = 0.10,
                          r2_gap_warn: float = 0.05) -> pd.DataFrame:
        """Return overfitting signals as a single-row DataFrame (Korean columns)."""
        if self.train_metric_at_best is None or self.valid_metric_at_best is None:
            raise ValueError("Overfitting signals not captured. Run fit() first.")

        metric = "rmse" if self.eval_metric_key == "rmse" else "r2"
        if metric == "rmse":
            warn_gap = (self.generalization_gap_pct_ is not None and self.generalization_gap_pct_ > rmse_gap_warn)
            warn_drift = (self.valid_drift_from_best_ is not None and self.valid_drift_from_best_ > 0.0)
        else:
            warn_gap = (self.generalization_gap_ is not None and self.generalization_gap_ > r2_gap_warn)
            warn_drift = (self.valid_drift_from_best_ is not None and self.valid_drift_from_best_ > 0.0)

        row = {
            "평가지표": metric.upper(),
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

    # ---------- SHAP & Feature Importance ----------
    def shap_importance_df(self, topk: Optional[int] = 30) -> Optional[pd.DataFrame]:
        """Return mean|SHAP| importance DataFrame on X_test (topk)."""
        if self.explainer is None or self.X_test is None:
            return None
        try:
            expl = self.explainer(self.X_test)
            vals = np.abs(np.asarray(expl.values)).mean(axis=0)
            cols = list(self.columns) if self.columns is not None else [f"f{i}" for i in range(len(vals))]
            df = pd.DataFrame({"feature": cols, "mean_abs_shap": vals}).sort_values("mean_abs_shap", ascending=False)
            if topk:
                df = df.head(topk)
            return df.reset_index(drop=True)
        except Exception:
            return None

    def catboost_importance_df(self, topk: Optional[int] = 30) -> Optional[pd.DataFrame]:
        """Return CatBoost native importance DataFrame."""
        if self.model is None or self.columns is None:
            return None
        try:
            imp = self.model.get_feature_importance()
            df = pd.DataFrame({"feature": list(self.columns), "importance": imp}).sort_values("importance", ascending=False)
            if topk:
                df = df.head(topk)
            return df.reset_index(drop=True)
        except Exception:
            return None

    # ---------- Save / Load ----------
    def _build_artifact(self, target: str, filename_pattern: str, include_shap: bool = True) -> Dict[str, Any]:
        """Build artifact dict to persist model and insights."""
        if not self.trained or self.model is None:
            raise ValueError("Model is not trained. Call fit() before save_model().")

        X = None if self.X_test is None else self.X_test.copy()
        y = None if self.y_test is None else self.y_test.copy()
        scores = self.evaluate() if (X is not None and y is not None) else {"r2": None, "rmse": None, "y_pred": None}
        overfit_row = self.overfit_report_df().iloc[0].to_dict()

        y_pred_test = None
        if X is not None:
            y_pred_test = self.predict(X)

        shap_values = None
        shap_base = None
        if include_shap and X is not None and self.explainer is not None:
            try:
                sh = self.explainer(X)
                shap_values = np.asarray(sh.values)
                base = sh.base_values
                shap_base = float(np.mean(base)) if np.ndim(base) > 0 else float(base)
            except Exception:
                shap_values = None
                shap_base = None

        return {
            "version": "v3-final",
            "timestamp": _now_iso(),
            "target": target,
            "filename_pattern": filename_pattern,
            "model_name": self.model_name,
            "model": self.model,
            "params": self.best_params or self.get_default_params(),
            "columns": list(self.columns) if self.columns is not None else None,
            "cat_features": self.cat_features,
            "ntree_end_": self.ntree_end_,
            "scores": {
                "r2": float(scores["r2"]) if scores["r2"] is not None else None,
                "rmse": float(scores["rmse"]) if scores["rmse"] is not None else None
            },
            "optuna_log": self.optuna_log if self.use_optuna else None,
            "learning_curve_": self.learning_curve_,
            "overfit_report_df": overfit_row,
            "X_test": X,
            "y_test": y,
            "y_pred_test": y_pred_test,
            "shap_values": shap_values,
            "shap_base_value": shap_base,
        }

    def save_model(self,
                   target: str,
                   filename_pattern: str,
                   protocol: Literal["binary", "ascii"] = "binary") -> str:
        """Persist model and insights to disk.

        Args:
            target: Identifier used in filename, e.g., 'task_y1'.
            filename_pattern: User-defined filename suffix.
            protocol: 'binary' (joblib) or 'ascii' (pickle).

        Returns:
            Absolute file path of saved artifact.
        """
        save_dir = "model/catboost"
        os.makedirs(save_dir, exist_ok=True)

        scores = self.scores()
        r2_tag = scores.get("r2", 0.0) if scores else 0.0
        target_col = target.split("_")[-1] if isinstance(target, str) else str(target)
        file_name = f"{target_col}_CAT_r2_{float(r2_tag):.4f}_n{self.n_trials}_{filename_pattern}.pkl"
        file_path = os.path.join(save_dir, file_name)

        artifact = self._build_artifact(target=target, filename_pattern=filename_pattern, include_shap=True)

        if protocol == "binary":
            joblib.dump(artifact, file_path)
        elif protocol == "ascii":
            with open(file_path, "wb") as f:
                pickle.dump(artifact, f)
        else:
            raise ValueError("protocol must be 'binary' or 'ascii'")

        return os.path.abspath(file_path)

    def load_model(self, path: str) -> "CatBoostRunner":
        """Load saved artifact and restore runner state.

        Args:
            path: Path to file saved by save_model().

        Returns:
            self
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No such file: {path}")

        try:
            artifact = joblib.load(path)
        except Exception:
            with open(path, "rb") as f:
                artifact = pickle.load(f)

        self.model_name = artifact.get("model_name", "CatBoost")
        self.model = artifact.get("model")
        self.columns = pd.Index(artifact.get("columns")) if artifact.get("columns") is not None else None
        self.cat_features = artifact.get("cat_features")
        self.ntree_end_ = artifact.get("ntree_end_")
        self.learning_curve_ = artifact.get("learning_curve_")
        self.optuna_log = artifact.get("optuna_log")
        self.best_params = artifact.get("params")

        self.X_test = artifact.get("X_test")
        self.y_test = artifact.get("y_test")
        self.trained = True

        # Cached extras (not strictly required for inference)
        self._loaded_overfit_dict = artifact.get("overfit_report_df", {})
        self._loaded_scores = artifact.get("scores", {})
        self._y_pred_test_loaded = artifact.get("y_pred_test", None)
        self._shap_values_loaded = artifact.get("shap_values", None)
        self._shap_base_value_loaded = artifact.get("shap_base_value", None)

        # Explainer is expensive; recreate lazily if needed
        self.explainer = None
        return self

    # ---------- Bundle Summary ----------
    def summarize_bundle(self) -> Dict[str, Any]:
        """Summarize loaded artifact metadata, scores and small preview."""
        if not self.trained or self.model is None:
            raise ValueError("Model not loaded or trained.")

        scores = self._loaded_scores if getattr(self, "_loaded_scores", None) else self.scores()
        of_dict = getattr(self, "_loaded_overfit_dict", None) or {}

        if getattr(self, "_y_pred_test_loaded", None) is not None:
            y_pred_preview = np.asarray(self._y_pred_test_loaded[:10]).tolist()
        elif self.X_test is not None:
            try:
                y_pred_preview = self.predict(self.X_test)[:10].tolist()
            except Exception:
                y_pred_preview = None
        else:
            y_pred_preview = None

        return {
            "model": self.model_name,
            "columns": list(self.columns) if self.columns is not None else None,
            "cat_features": self.cat_features,
            "ntree_end": self.ntree_end_,
            "scores": scores,
            "overfit_report": of_dict,
            "learning_curve_keys": list(self.learning_curve_.keys()) if self.learning_curve_ else None,
            "X_test_shape": list(self.X_test.shape) if self.X_test is not None else None,
            "y_test_len": int(len(self.y_test)) if self.y_test is not None else None,
            "y_pred_test_preview": y_pred_preview,
            "has_shap_values": bool(getattr(self, "_shap_values_loaded", None) is not None),
        }

    # ---------- Learning Curve Plot ----------
    def plot_learning_curve(self) -> Optional[go.Figure]:
        """Plot train/valid learning curves if available."""
        lc = self.learning_curve_
        if not lc or "train" not in lc or "valid" not in lc:
            return None
        x = list(range(1, len(lc["train"]) + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=lc["train"], mode="lines", name="Train"))
        fig.add_trace(go.Scatter(x=x, y=lc["valid"], mode="lines", name="Valid"))
        fig.update_layout(
            title=f"[{self.model_name}] Learning Curve",
            xaxis_title="Iteration",
            yaxis_title=self._metric_name(),
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig
