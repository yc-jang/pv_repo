# report_models.ipynb / (또는 .py) 셀에 붙여넣어 사용
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from IPython.display import display, Markdown

# 환경에 맞게 조정
from model_cat import load_and_inspect   # v1 (runner v5 기반)
from runner import CatBoostRunner        # v5

# =========================
# Config
# =========================
DEPLOY_DIR = Path("models/deploy")
MODEL_EXTS = (".pkl", ".joblib")  # 필요시 확장자 추가
TOPK = 20                          # shap_summary_bar / waterfall 공통 topk


# =========================
# Utilities
# =========================
def find_models(deploy_dir: Path, exts: Tuple[str, ...]) -> List[Path]:
    """Find model artifact files under deploy_dir."""
    paths: List[Path] = []
    if not deploy_dir.exists():
        raise FileNotFoundError(f"Deploy directory not found: {deploy_dir}")
    for p in deploy_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            paths.append(p)
    if not paths:
        raise FileNotFoundError(f"No model files with {exts} under {deploy_dir}")
    return sorted(paths)


def load_metadata_csv(deploy_dir: Path) -> Optional[pd.DataFrame]:
    """Load optional metadata.csv to enrich '학습 파일 정보'."""
    meta_path = deploy_dir / "metadata.csv"
    if meta_path.exists():
        try:
            df = pd.read_csv(meta_path)
            # 권장 컬럼 예시: filename, train_data_path, created_by, created_at, notes...
            return df
        except Exception:
            return None
    return None


def md_title(txt: str, level: int = 2) -> None:
    """Display a Markdown title."""
    display(Markdown("#" * level + f" {txt}"))


def safe_preview(df: Optional[pd.DataFrame], n: int = 5) -> Optional[pd.DataFrame]:
    """Return a small preview of a DataFrame (or None)."""
    if df is None:
        return None
    try:
        return df.head(n)
    except Exception:
        return None


# =========================
# Section Renderers
# =========================
def render_training_file_info(runner: CatBoostRunner,
                              bundle_summary: Dict[str, Any],
                              meta_row: Optional[pd.Series]) -> None:
    """학습 파일/환경 관련 정보 섹션."""
    md_title("1) 학습에 사용된 파일/환경 정보", level=3)

    rows = []

    # metadata.csv 우선
    if meta_row is not None:
        for k, v in meta_row.items():
            rows.append(("metadata." + str(k), v))

    # runner/bundle에서 유의미한 것들
    rows.extend([
        ("model_name", bundle_summary.get("model")),
        ("timestamp", bundle_summary.get("timestamp", "N/A")),
        ("columns(count)", len(bundle_summary.get("columns") or [])),
        ("cat_features", bundle_summary.get("cat_features")),
        ("ntree_end", bundle_summary.get("ntree_end")),
        ("params(best)", runner.best_params),
    ])

    info_df = pd.DataFrame(rows, columns=["key", "value"])
    display(info_df)


def render_test_info(runner: CatBoostRunner, bundle_summary: Dict[str, Any]) -> None:
    """테스트 데이터 정보 섹션."""
    md_title("2) 테스트 데이터 정보", level=3)

    x_shape = bundle_summary.get("X_test_shape")
    y_len = bundle_summary.get("y_test_len")
    rows = [
        ("X_test_shape", x_shape),
        ("y_test_len", y_len),
    ]
    info_df = pd.DataFrame(rows, columns=["key", "value"])
    display(info_df)

    # 미리보기(가능한 경우)
    x_prev = safe_preview(runner.X_test, n=5)
    y_prev = safe_preview(runner.y_test.to_frame(name="y_test") if runner.y_test is not None else None, n=5)

    if x_prev is not None:
        display(Markdown("**X_test preview (head)**"))
        display(x_prev)
    if y_prev is not None:
        display(Markdown("**y_test preview (head)**"))
        display(y_prev)


def render_training_results(runner: CatBoostRunner,
                            scores: Dict[str, float],
                            overfit_report_df: Optional[pd.DataFrame],
                            figures: Dict[str, Any]) -> None:
    """학습 결과 섹션: 점수/오버핏/러닝커브/예측-정답."""
    md_title("3) 학습 결과", level=3)

    # 점수 테이블
    score_df = pd.DataFrame([scores])
    display(Markdown("**Scores (test)**"))
    display(score_df)

    # 오버피팅 리포트
    if overfit_report_df is not None:
        display(Markdown("**Overfitting Report**"))
        display(overfit_report_df)

    # 예측 vs 정답 / 산점도 / 잔차 / 러닝커브
    if figures.get("line") is not None:
        display(Markdown("**예측 vs 정답 (Line)**"))
        figures["line"].show()

    if figures.get("pred_vs_true") is not None:
        display(Markdown("**예측 vs 정답 (Scatter)**"))
        figures["pred_vs_true"].show()

    if figures.get("residuals") is not None:
        display(Markdown("**잔차 산점도**"))
        figures["residuals"].show()

    if figures.get("learning_curve") is not None:
        display(Markdown("**Learning Curve**"))
        figures["learning_curve"].show()


def render_shap(runner: CatBoostRunner, topk: int = 20) -> None:
    """SHAP 섹션: 요약바 + 글로벌 순서 워터폴."""
    md_title("4) SHAP", level=3)

    # Top-k summary bar
    imp_df, imp_fig = runner.shap_summary_bar(topk=topk)
    display(Markdown(f"**SHAP Importance (Top {topk})**"))
    display(imp_df)
    if imp_fig is not None:
        imp_fig.show()

    # Waterfall (전역 중요도 순서 고정, idx=None → 중앙값 예측 샘플)
    # v5의 plot_waterfall_with_mean 구현 사용
    wf_df, wf_fig = runner.plot_waterfall_with_mean(idx=None, topk=topk, show=False)
    display(Markdown("**SHAP Waterfall (global-order, median-pred sample)**"))
    display(wf_df)
    wf_fig.show()


# =========================
# Main Orchestrator
# =========================
def build_and_render_report(deploy_dir: Path = DEPLOY_DIR,
                            exts: Tuple[str, ...] = MODEL_EXTS,
                            topk: int = TOPK) -> None:
    """models/deploy 하단 모델들을 일괄 로드하고, 노트북에 섹션별로 출력한다.

    흐름: 학습파일(메타) → 테스트 정보 → 학습결과 → SHAP
    """
    paths = find_models(deploy_dir, exts)
    meta_df = load_metadata_csv(deploy_dir)

    display(Markdown(f"## 📦 Model Report — {len(paths)} artifact(s) in `{deploy_dir}`"))

    for p in paths:
        display(Markdown(f"---\n### 📁 {p.name}"))
        # metadata join (filename 키 권장)
        meta_row = None
        if meta_df is not None:
            m = meta_df[meta_df["filename"].astype(str) == p.name]
            if len(m) > 0:
                meta_row = m.iloc[0]

        # 1) 로드 & 인스펙트
        out = load_and_inspect(str(p))
        runner: CatBoostRunner = out["runner"]
        bundle_summary: Dict[str, Any] = out["bundle_summary"]
        scores: Dict[str, float] = out.get("runner").scores()
        of_df = out.get("bundle_summary", {}).get("overfit_report")  # dict일 수 있음
        # dict인 경우 DF로
        overfit_df = pd.DataFrame([of_df]) if isinstance(of_df, dict) else out.get("overfit_report_df", None)

        figures = out.get("figures", {})

        # 2) 섹션 렌더링
        render_training_file_info(runner, bundle_summary, meta_row)
        render_test_info(runner, bundle_summary)
        render_training_results(runner, scores, overfit_df, figures)
        render_shap(runner, topk=topk)


# =========================
# Run (in notebook)
# =========================
# build_and_render_report(DEPLOY_DIR, MODEL_EXTS, TOPK)
