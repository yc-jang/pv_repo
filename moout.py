from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

def _ensure_xlsx_path(path: str | Path) -> Path:
    p = Path(path)
    if p.suffix.lower() != ".xlsx":
        raise ValueError(f"엑셀 경로는 .xlsx 여야 합니다: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)  # 상위 폴더 자동 생성
    return p

def export_df_to_excel(df: pd.DataFrame, output_path: str | Path, sheet_name: str = "predictions") -> Path:
    """DataFrame을 지정된 .xlsx 경로에 저장(폴더 자동 생성)."""
    out = _ensure_xlsx_path(output_path)
    # 항상 덮어쓰기; 필요 시 mode/if_sheet_exists 옵션 확장 가능
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=True, sheet_name=sheet_name)
    return out

def build_model_list(cat_model, n_models: int = 7, model_names: list[str] | None = None):
    """
    cat_model 하나를 n_models개로 '복제 사용'하는 리스트 구성.
    실제 복제(copy)가 필요 없다면 동일 객체 참조로 충분.
    """
    if model_names is None:
        model_names = [f"model_{i+1}" for i in range(n_models)]
    if len(model_names) != n_models:
        raise ValueError("model_names 길이가 n_models와 다릅니다.")
    models = [(name, cat_model) for name in model_names]  # 동일 모델 재사용
    return models

def predict_models_to_df(models: list[tuple[str, object]], X: pd.DataFrame) -> pd.DataFrame:
    """
    여러 모델의 예측을 하나의 DataFrame으로 반환.
    - X.index를 그대로 유지
    - 단일 모델이어도 DataFrame 형식 보장 (1열)
    """
    preds = {}
    for name, model in models:
        y = model.predict(X)  # CatBoost의 predict 결과
        y = np.asarray(y).reshape(-1)  # 1차원 보장
        if len(y) != len(X.index):
            raise ValueError(f"{name} 예측 길이({len(y)})와 X 행수({len(X)}) 불일치")
        preds[name] = y
    out_df = pd.DataFrame(preds, index=X.index)
    return out_df

def run_and_export(cat_model, X: pd.DataFrame, output_xlsx: str | Path,
                   n_models: int = 7, model_names: list[str] | None = None,
                   sheet_name: str = "predictions") -> Path:
    """
    - cat_model을 n_models개로 복제 사용하여 예측
    - X.index 유지 + 항상 DataFrame 반환
    - 지정된 엑셀(output_xlsx)에 저장 보장(폴더 자동 생성)
    - 최종 저장 경로 반환
    """
    models = build_model_list(cat_model, n_models=n_models, model_names=model_names)
    pred_df = predict_models_to_df(models, X)
    return export_df_to_excel(pred_df, output_xlsx, sheet_name=sheet_name)
