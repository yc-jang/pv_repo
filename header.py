import pandas as pd
import numpy as np

def flatten_multi_header(df: pd.DataFrame, sep: str = "_") -> pd.DataFrame:
    """
    엑셀의 가로/세로 병합이 포함된 다중 헤더를 단일 문자열로 완벽하게 평탄화합니다.
    레벨의 깊이(Depth)나 숫자형 헤더 혼재 여부와 관계없이 항상 동작합니다.
    """
    # [Guard] 이미 단일 인덱스면 즉시 반환
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    # 1. MultiIndex -> DataFrame (행: 헤더 레벨, 열: 컬럼 인덱스)
    header_df = pd.DataFrame(df.columns.to_list()).T

    # -------------------------------------------------------------------------
    # [Patch] 헤더에 정수/실수가 섞여 있을 때 정규식(replace) 에러 방지
    # -------------------------------------------------------------------------
    header_df = header_df.astype(str)

    # 2. 엑셀 병합으로 생긴 가짜 값들을 진짜 NaN으로 통일
    # (?i) 대소문자 무시 / Unnamed 시작 / nan, None 문자열 / 아예 빈칸
    pattern = r'(?i)^Unnamed:.*|^nan$|^None$|^$'
    header_df = header_df.replace(pattern, np.nan, regex=True)

    # 3. [가로 병합 해결] axis=1 방향으로 빈칸을 앞의 값으로 채움
    header_df = header_df.ffill(axis=1)

    new_columns = []
    
    for col_idx in header_df.columns:
        # 4. [세로 병합 해결] NaN 값들을 완전히 제거
        valid_vals = header_df[col_idx].dropna().tolist()
        
        # 5. [중복 제거 로직] 상하위 병합 시 'ID'_'ID' 형태로 붙는 것을 방지
        dedup_vals = []
        for val in valid_vals:
            # 리스트가 비어있거나, 직전 값과 다를 때만 추가 (연속 중복 제거)
            if not dedup_vals or dedup_vals[-1] != val:
                dedup_vals.append(val)
                
        # 6. 최종 문자열 조합 (만약 모든 값이 다 날아갔다면 임시 이름 부여)
        new_col_name = sep.join(dedup_vals) if dedup_vals else f"Column_{col_idx}"
        new_columns.append(new_col_name)

    # 7. 원본 보호를 위해 복사본에 새 컬럼 적용 후 반환
    df_result = df.copy()
    df_result.columns = new_columns
    
    return df_result
