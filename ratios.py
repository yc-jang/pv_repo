import pandas as pd
import numpy as np

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    전구체 및 원료 물성 데이터를 기반으로 파생 변수를 생성하고,
    다중공선성이 예상되는 변수는 제거하여 모델 예측 성능 및 안정성을 높입니다.

    Args:
        df (pd.DataFrame): 원료 물성 및 화학조성 정보가 포함된 데이터프레임

    Returns:
        pd.DataFrame: 파생 변수 생성 및 다중공선성 변수 제거가 완료된 데이터프레임
    """
    df = df.copy()  # 원본 훼손 방지

    # 원료 물질 리스트
    materials = ['N86', 'LIOH', 'ALOH', 'ZRO2']
    impurities = ['Na', 'S', 'Ca', 'Cl']

    # ===== 1. 입도 기반 파생 변수 (span index) =====
    # span = (D90 - D10) / D50
    for mat in materials:
        d10_col = f'물성_입도_D10_{mat}'
        d50_col = f'물성_입도_D50_{mat}'
        d90_col = f'물성_입도_D90_{mat}'
        if all(col in df.columns for col in [d10_col, d50_col, d90_col]):
            df[f'span_index_{mat}'] = (df[d90_col] - df[d10_col]) / df[d50_col]

    # ===== 2. 조성 기반 파생 변수 =====
    # Ni 비율: capacity에 큰 영향 → 유지
    try:
        df['Ni_ratio_N86'] = df['화학성분_Ni_N86'] / (
            df['화학성분_Ni_N86'] + df['화학성분_Co_N86'] + df['화학성분_Mn_N86']
        )
    except KeyError:
        pass

    # Li_excess: Li가 전이금속 총합보다 많을 경우 반응성과 용량이 증가하는 경우 고려
    try:
        df['Li_excess'] = df['화학성분_Li_LIOH'] - (
            df.get('화학성분_Ni_N86', 0) +
            df.get('화학성분_Co_N86', 0) +
            df.get('화학성분_Mn_N86', 0)
        )
    except KeyError:
        pass

    # ===== 3. 불순물 log 변환 =====
    # Impurity는 ppm 단위로 편차가 크므로 log1p로 스케일 안정화
    for ele in impurities:
        for mat in materials:
            raw_col = f'물성_불순물_{ele}_{mat}'
            log_col = f'log_impurity_{ele}_{mat}'
            if raw_col in df.columns:
                df[log_col] = np.log1p(df[raw_col])

    # ===== 4. BET 비선형 파생 =====
    # BET^2는 비선형성 반영 목적. 단, 기본 BET와 중복성이 크므로 나중에 제거 가능
    for mat in materials:
        bet_col = f'물성_BET_{mat}'
        if bet_col in df.columns:
            df[f'BET2_{mat}'] = df[bet_col] ** 2

    # ===== 5. 다중공선성 제거 =====
    drop_cols = []

    # (1) 입도: D10, D90은 span 계산에 포함되므로 별도 보존 필요 없음
    for mat in materials:
        for d in ['D10', 'D90']:
            col = f'물성_입도_{d}_{mat}'
            if col in df.columns:
                drop_cols.append(col)

    # (2) 불순물: log 변환본이 있으므로 원본 제거
    for ele in impurities:
        for mat in materials:
            raw_col = f'물성_불순물_{ele}_{mat}'
            log_col = f'log_impurity_{ele}_{mat}'
            if raw_col in df.columns and log_col in df.columns:
                drop_cols.append(raw_col)

    # (3) BET^2는 과적합 또는 공선성 가능성으로 제거 (원본 BET 유지)
    for mat in materials:
        bet = f'물성_BET_{mat}'
        bet2 = f'BET2_{mat}'
        if bet in df.columns and bet2 in df.columns:
            drop_cols.append(bet2)

    # (4) Ni 비율이 있다면 원본 Ni 농도 제거
    if '화학성분_Ni_N86' in df.columns and 'Ni_ratio_N86' in df.columns:
        drop_cols.append('화학성분_Ni_N86')

    # 실제 제거
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    return df
