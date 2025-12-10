from __future__ import annotations

from typing import Dict, Iterable
import numpy as np
import pandas as pd


def setlot_value_function(
    self,
    lot_to_setlot_dict: Dict[str, Iterable[str]],
    fill_df: pd.DataFrame,
    fill_lot_col: str,
) -> pd.DataFrame:
    """LOT 그룹별 1·2·3단 'Saga당 충진량' 평균을 계산해 테이블로 반환한다.

    각 기준 LOT(key)에 대해 lot_to_setlot_dict[key]에 포함된 LOT들을
    fill_df[fill_lot_col]에서 찾아, 해당 행들의 1단/2단/3단 'Saga당 충진량' 평균을 계산한다.

    충진량 및 Saga 수 컬럼은 문자열(공백·쉼표 포함)일 수 있으므로, float으로 정제한 뒤
    (충진량 / Saga 수)로 변환한 값을 기준으로 평균을 구한다.
    매칭되는 행이 없거나 Saga 수가 0인 경우 NaN이 되며, 최종적으로 0.0으로 대체된다.

    Args:
        lot_to_setlot_dict (Dict[str, Iterable[str]]):
            기준 LOT → 함께 평균을 낼 LOT들의 목록 매핑.
        fill_df (pd.DataFrame):
            충진량 정보와 LOT 컬럼(fill_lot_col)을 포함한 원본 DataFrame.
            반드시 다음 컬럼들을 포함해야 한다:
                - '1단_충진량', '2단_충진량', '3단_충진량'
                - '1단_Saga 수', '2단_Saga 수', '3단_Saga 수'
        fill_lot_col (str):
            fill_df에서 LOT를 나타내는 컬럼 이름.

    Returns:
        pd.DataFrame:
            인덱스가 기준 LOT, 컬럼이 1단/2단/3단 'Saga당 충진량' 평균인 DataFrame.
    """
    df = fill_df.copy()

    # 단별 충진량 컬럼 / Saga 수 컬럼 정의
    charge_cols = ["1단_충진량", "2단_충진량", "3단_충진량"]
    saga_cols = ["1단_Saga 수", "2단_Saga 수", "3단_Saga 수"]

    # 충진량 컬럼 정제: 문자열 → 공백/콤마 제거 → float
    for col in charge_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(",", "")
            .astype(float)
        )

    # Saga 수 컬럼 정제: 문자열 → 공백/콤마 제거 → float
    for col in saga_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(",", "")
            .astype(float)
        )

    # 각 단별로 'Saga당 충진량'으로 변환: 충진량 / Saga 수
    # Saga 수가 0이면 NaN으로 처리 → 이후 평균 계산 시 자동 제외, 마지막에 0.0으로 채움
    for charge_col, saga_col in zip(charge_cols, saga_cols):
        saga = df[saga_col].replace(0, np.nan)
        df[charge_col] = df[charge_col] / saga

    # 기준 LOT별 평균 'Saga당 충진량' Series를 딕셔너리로 수집
    lot_to_charge: Dict[str, pd.Series] = {}
    for lot_key, setlot_group in lot_to_setlot_dict.items():
        mask = df[fill_lot_col].isin(setlot_group)
        lot_to_charge[lot_key] = df.loc[mask, charge_cols].mean()

    # 딕셔너리 → DataFrame 변환, NaN은 0.0으로 대체
    result = (
        pd.DataFrame.from_dict(lot_to_charge, orient="index")
        .fillna(0.0)
    )

    return result
