from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

def _ensure_xlsx_path(path: str | Path) -> Path:
    p = Path(path)
    if p.suffix.lower() != ".xlsx":
        raise ValueError(f"엑셀 경로는 .xlsx 여야 합니다: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def _prepare_X_for_model(end2end_data: pd.DataFrame, model) -> pd.DataFrame:
    """
    end2end_data의 컬럼을 model.X_test.columns 순서로 정렬.
    - end2end_data가 필요한 컬럼을 모두 포함하는지 확인
    - 추가 컬럼은 무시(사용 안함)
    """
    if not hasattr(model, "X_test") or not hasattr(model.X_test, "columns"):
        raise AttributeError("model.X_test.columns 를 찾을 수 없습니다.")
    req_cols = list(model.X_test.columns)

    # 누락 컬럼 검사
    missing = [c for c in req_cols if c not in end2end_data.columns]
    if missing:
        raise ValueError(f"입력 데이터에 누락된 컬럼: {missing}")

    # 모델이 요구하는 순서로 재정렬
    X_ordered = end2end_data[req_cols].copy()
    # (선택) 타입/NaN 전처리가 필요하면 여기서 추가
    return X_ordered

def predict_cat_to_seven(cat_model, end2end_data: pd.DataFrame) -> pd.DataFrame:
    """
    - cat_model 기준으로 컬럼 순서 확인/정렬
    - cat_model 예측 1개를 7개 컬럼으로 복제
    - index는 end2end_data의 index를 그대로 사용
    - 항상 (n_rows x 7) DataFrame 반환
    """
    X = _prepare_X_for_model(end2end_data, cat_model)
    y = np.asarray(cat_model.predict(X)).reshape(-1)
    if len(y) != len(X.index):
        raise ValueError("예측 길이와 입력 행수가 일치하지 않습니다.")

    out = pd.DataFrame(
        {f"model_{i}": y for i in range(1, 8)},
        index=X.index  # 입력 인덱스 유지
    )
    return out
https://github.com/yc-jang/pv_repo/pull/10
def export_predictions_to_excel(df: pd.DataFrame, model_output_path: str | Path) -> Path:
    """
    결과 DataFrame을 지정된 엑셀 경로의 Sheet1에 저장(덮어쓰기).
    """
    out = _ensure_xlsx_path(model_output_path)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Sheet1", index=True)
    return out

# ========= 사용 예 =========
# pred_df = predict_cat_to_seven(cat_model, end2end_data)
# final_path = export_predictions_to_excel(pred_df, "C:/result/model_output.xlsx")
# print("저장 완료:", final_path)


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
