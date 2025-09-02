"""
model_cat.py

CatBoostRunner(v3) 사용 예시 파이프라인:
- 데이터 분할 → 학습 → 평가 → 오버피팅 리포트 → 저장
- 저장된 번들 로드 → 요약/결과/플롯/비교용 DF 생성

요구사항 반영:
- 함수별 type hints, Google 스타일 docstring
- 예외는 꼭 필요한 경우에만 발생
- 불필요한 로깅 제거
- plotly 반환(표준 결과/플롯은 Runner가 제공)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 경로/패키지에 맞게 조정하세요.
from catboost_runner import CatBoostRunner  # v3 최종본


# =========================
# Config
# =========================
@dataclass
class CatConfig:
    """CatBoost 파이프라인 설정값.

    Attributes:
        target_col: 타깃 컬럼명.
        use_optuna: Optuna 탐색 사용 여부.
        n_trials: Optuna trial 횟수.
        eval_metric: 'rmse' 또는 'r2'.
        filename_pattern: 저장 파일명 패턴(접미사).
        id_col: 결과 DF에 식별자로 포함할 컬럼(선택).
        test_size: 홀드아웃 테스트셋 비율.
        random_state: 데이터 분할 시드.
    """
    target_col: str
    use_optuna: bool = True
    n_trials: int = 30
    eval_metric: str = "rmse"
    filename_pattern: str = "exp001"
    id_col: Optional[str] = None
    test_size: float = 0.2
    random_state: int = 42


# =========================
# Pipeline Helpers
# =========================
def split_xy(df: pd.DataFrame, target_col: str, test_size: float = 0.2,
            random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """피처/타깃 분리 후 학습/평가 데이터 분할.

    Args:
        df: 원본 데이터프레임.
        target_col: 타깃 컬럼명.
        test_size: 테스트셋 비율.
        random_state: 랜덤 시드.

    Returns:
        X_train, X_test, y_train, y_test
    """
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in DataFrame.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def build_runner(cfg: CatConfig) -> CatBoostRunner:
    """설정값으로 CatBoostRunner 인스턴스를 생성한다.

    Args:
        cfg: CatConfig

    Returns:
        CatBoostRunner
    """
    runner = CatBoostRunner(
        use_optuna=cfg.use_optuna,
        n_trials=cfg.n_trials,
        eval_metric=cfg.eval_metric,
        cat_features=None  # 필요시 ['col_a','col_b'] 등 명시 지정
    )
    return runner


def train_evaluate_save(df: pd.DataFrame, cfg: CatConfig) -> Dict[str, Any]:
    """학습→평가→오버피팅 리포트→저장까지 한 번에 수행.

    Args:
        df: 전체 데이터프레임(타깃 포함).
        cfg: 파이프라인 설정.

    Returns:
        결과 딕셔너리(점수, 리포트DF, 결과DF, 저장경로 등).
    """
    X_train, X_test, y_train, y_test = split_xy(
        df, target_col=cfg.target_col, test_size=cfg.test_size, random_state=cfg.random_state
    )

    # Runner 구성 및 데이터 주입
    runner = build_runner(cfg)
    runner._update_data(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    # 학습
    runner.fit()

    # 평가/리포트
    scores = runner.evaluate()  # {'r2':..., 'rmse':..., 'y_pred':...}
    of_df = runner.overfit_report_df()  # 한국어 컬럼 1-row DF
    res_long = runner.get_results(wide=False, id_col=cfg.id_col)  # 표준 long DF
    res_wide = runner.get_results(wide=True, id_col=cfg.id_col)   # y_pred_<model>

    # 모델 저장
    save_path = runner.save_model(
        target=f"task_{cfg.target_col}",
        filename_pattern=cfg.filename_pattern,
        protocol="binary"
    )

    # (선택) 플롯: 호출만 해도 Figure 반환됨 (표시/저장은 사용자 환경에서)
    fig_line = runner.plot_predictions_line(sample=None)
    fig_scatter = runner.plot_pred_vs_true_scatter()
    fig_resid = runner.plot_residuals_scatter()
    fig_lc = runner.plot_learning_curve()

    return {
        "runner": runner,
        "scores": {"r2": scores["r2"], "rmse": scores["rmse"]},
        "overfit_report_df": of_df,
        "results_long": res_long,
        "results_wide": res_wide,
        "save_path": save_path,
        "figures": {
            "line": fig_line,
            "pred_vs_true": fig_scatter,
            "residuals": fig_resid,
            "learning_curve": fig_lc,
        },
    }


def load_and_inspect(path: str, id_col: Optional[str] = None) -> Dict[str, Any]:
    """저장된 번들을 로드해 요약/결과/플롯 등을 반환.

    Args:
        path: save_model로 저장된 파일 경로.
        id_col: 결과 DF에 포함할 식별자 컬럼(선택, X_test에 존재해야 함).

    Returns:
        번들 요약, 결과 DF, 플롯 객체 등을 담은 딕셔너리.
    """
    runner = CatBoostRunner().load_model(path)
    info = runner.summarize_bundle()

    res_long = runner.get_results(wide=False, id_col=id_col)
    res_wide = runner.get_results(wide=True, id_col=id_col)

    fig_line = runner.plot_predictions_line(sample=None)
    fig_scatter = runner.plot_pred_vs_true_scatter()
    fig_resid = runner.plot_residuals_scatter()
    fig_lc = runner.plot_learning_curve()

    # 중요도 DF (가능할 때만)
    shap_imp = runner.shap_importance_df(topk=30)
    cat_imp = runner.catboost_importance_df(topk=30)

    return {
        "runner": runner,
        "bundle_summary": info,
        "results_long": res_long,
        "results_wide": res_wide,
        "figures": {
            "line": fig_line,
            "pred_vs_true": fig_scatter,
            "residuals": fig_resid,
            "learning_curve": fig_lc,
        },
        "importances": {
            "shap": shap_imp,
            "catboost": cat_imp,
        },
    }


# =========================
# Example main
# =========================
if __name__ == "__main__":
    # 예시 데이터 생성 (실전에서는 실제 df로 교체)
    rng = np.random.default_rng(42)
    n = 1000
    df_example = pd.DataFrame({
        "f1": rng.normal(size=n),
        "f2": rng.normal(loc=1.0, scale=2.0, size=n),
        "f3": rng.integers(0, 5, size=n).astype("category"),
        "y": 3.0 + 2.0 * rng.normal(size=n) + rng.normal(scale=0.5, size=n),
    })

    cfg = CatConfig(
        target_col="y",
        use_optuna=False,    # 데모에서는 빠르게
        n_trials=20,
        eval_metric="rmse",
        filename_pattern="demo",
        id_col=None,
        test_size=0.2,
        random_state=42,
    )

    # 학습/저장
    out = train_evaluate_save(df_example, cfg)
    print("Saved to:", out["save_path"])
    print("Scores:", out["scores"])
    print(out["overfit_report_df"])

    # 로드/점검
    loaded = load_and_inspect(out["save_path"])
    print("Bundle summary:", loaded["bundle_summary"])
    print(loaded["results_long"].head())
