import pandas as pd
import numpy as np

def flatten_multi_header(df: pd.DataFrame, sep: str = "_") -> pd.DataFrame:
    """
    엑셀의 가로/세로 병합이 섞여 있는 복잡한 다중 헤더를 평탄화합니다.
    상위 레벨(Parent Level)의 경계를 인식하여 잘못된 값이 밀려오는 것을 완벽히 차단합니다.
    """
    # [Guard] 이미 단일 인덱스면 즉시 반환
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    # 1. MultiIndex -> DataFrame (행: 레벨, 열: 컬럼 인덱스)
    header_df = pd.DataFrame(df.columns.to_list()).T

    # 숫자형 헤더 에러 방지를 위한 문자열 변환 및 가짜 값(Unnamed 등) NaN 처리
    header_df = header_df.astype(str)
    pattern = r'(?i)^Unnamed:.*|^nan$|^None$|^$'
    header_df = header_df.replace(pattern, np.nan, regex=True)

    n_levels, n_cols = header_df.shape

    # -------------------------------------------------------------------------
    # [핵심 로직] 계층적 가로 채우기 (Hierarchical Forward Fill)
    # -------------------------------------------------------------------------
    for i in range(n_levels):  # 위쪽 레벨(0)부터 아래쪽 레벨로 내려가며 처리
        for j in range(n_cols): # 왼쪽(0)부터 오른쪽으로 이동
            
            # 현재 셀이 비어있다면 왼쪽 값으로 채울지 말지 결정해야 함
            if pd.isna(header_df.iloc[i, j]):
                
                if j > 0: # 첫 번째 열이 아닌 경우에만
                    # 조건: "내 위에 있는 모든 부모 레벨이 왼쪽 열의 부모 레벨과 똑같은가?"
                    if i == 0:
                        parents_match = True  # 최상위 레벨은 부모가 없으므로 무조건 무통과
                    else:
                        left_parents = header_df.iloc[:i, j-1]
                        curr_parents = header_df.iloc[:i, j]
                        # Series.equals()는 두 그룹의 값이 완전히 같은지 검사
                        parents_match = left_parents.equals(curr_parents)

                    # 부모가 같을 때(진짜 가로 병합)만 왼쪽 값을 가져옴
                    if parents_match:
                        header_df.iloc[i, j] = header_df.iloc[i, j-1]
                        
                    # 부모가 다르면(세로 병합 등 경계선) 가져오지 않고 NaN 유지 (문제 해결 포인트!)

    # -------------------------------------------------------------------------
    # [이름 조합] 세로 병합 및 중복 처리
    # -------------------------------------------------------------------------
    new_columns = []
    for j in range(n_cols):
        # NaN 값들(세로 병합으로 비어있는 곳)을 제거
        valid_vals = header_df.iloc[:, j].dropna().tolist()
        
        # 'ID'_'ID' 처럼 상하위 이름이 중복되는 경우 제거
        dedup_vals = []
        for val in valid_vals:
            if not dedup_vals or dedup_vals[-1] != val:
                dedup_vals.append(val)
                
        # 문자열 조합 및 빈 컬럼 방어
        new_col_name = sep.join(dedup_vals) if dedup_vals else f"Column_{j}"
        new_columns.append(new_col_name)

    # 원본 보호를 위해 복사본에 새 컬럼 적용 후 반환
    df_result = df.copy()
    df_result.columns = new_columns
    
    return df_result


import pandas as pd
from typing import Dict, Set, Iterable

def build_lot_to_setlot_mapping(
    self, 
    df_target_lots: pd.DataFrame, 
    source_to_target_map: Dict[str, Iterable[str]], 
    target_lot_col: str, 
    setlot_col: str
) -> Dict[str, Set[str]]:
    """
    Source Lot을 기반으로 연관된 유효 SetLot(NaN 제외)들의 집합(Set)을 매핑하여 반환합니다.
    """
    
    # 1. [최적화 핵심] 사전에 NaN 제거 (Fail-Fast & Pre-filtering)
    # 나중에 하나씩 isna()로 검사할 필요 없이, 처음부터 SetLot이 없는 행을 날려버립니다.
    df_valid = df_target_lots.dropna(subset=[setlot_col])
    
    # 2. [중간 매핑] Target Lot -> {SetLot1, SetLot2, ...}
    # dropna를 거쳤으므로 이 딕셔너리의 value(set)에는 절대 NaN이 들어가지 않습니다.
    target_to_setlots = df_valid.groupby(target_lot_col)[setlot_col].apply(set).to_dict()
    
    # 3. [최종 병합] Source Lot -> {SetLot1, SetLot2, ...} (Fast Set Union)
    source_to_setlot_map = {}
    for source_lot, target_lots in source_to_target_map.items():
        
        # set().union()은 파이썬 내부 C단에서 처리되므로 리스트 sum()보다 압도적으로 빠릅니다.
        # .get(t, set())을 사용하여 target_to_setlots에 키가 없을 경우 빈 set을 반환해 에러를 방지합니다.
        combined_setlots = set().union(
            *(target_to_setlots.get(t, set()) for t in target_lots)
        )
        
        source_to_setlot_map[source_lot] = combined_setlots
        
    return source_to_setlot_map

