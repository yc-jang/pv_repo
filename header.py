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


import pandas as pd
from typing import Dict, Set

def compute_total_stop_time_df(heat_to_bake_map: Dict[str, Set[str]], stop_file: pd.DataFrame) -> pd.DataFrame:
    """
    heat_to_bake_map의 각 키(Heat)에 할당된 생산Lot번호들의 
    경과시간(분)을 모두 합산하고, 전체 총합을 포함한 데이터프레임으로 반환합니다.
    """
    result_data = []
    
    for heat, lot_set in heat_to_bake_map.items():
        # 1. stop_file에서 '생산Lot번호'가 현재 lot_set에 포함되는 행만 필터링
        mask = stop_file['생산Lot번호'].isin(lot_set)
        
        # 2. 필터링된 행들의 '경과시간(분)'을 모두 더함 (.sum()은 기본적으로 NaN을 0으로 취급하여 안전함)
        total_time = stop_file.loc[mask, '경과시간(분)'].sum()
        
        # 3. 결과를 리스트에 저장
        result_data.append({
            'Heat': heat,
            '총 정지시간(분)': total_time
        })
        
    # 4. 리스트를 데이터프레임으로 변환
    df_result = pd.DataFrame(result_data)
    
    # 5. 전체 총합 계산 및 하단에 행 추가
    if not df_result.empty:
        grand_total = df_result['총 정지시간(분)'].sum()
        total_row = pd.DataFrame({
            'Heat': ['[전체 총합]'],
            '총 정지시간(분)': [grand_total]
        })
        df_result = pd.concat([df_result, total_row], ignore_index=True)
        
    return df_result

import pandas as pd
import re

def read_clipboard_robustly(sep: str = '\t', force_rename_duplicates: bool = False) -> pd.DataFrame:
    """
    클립보드 데이터를 읽어오면서 데이터 유실 없이 원본 CSV 형태를 복원하는 함수.
    
    Args:
        sep (str): 구분자. 엑셀/그리드 복사 시 '\t', 쉼표 구분 텍스트 복사 시 ','
        force_rename_duplicates (bool): True일 경우 Pandas가 강제로 붙인 '.1', '.2' 접미사를 강제로 제거함.
                                        (주의: 컬럼명이 중복되면 이후 데이터 처리에 문제가 생길 수 있음)
    """
    try:
        # 1. 클립보드에서 데이터 읽기 (원본 텍스트의 앞뒤 공백 스트립 등 기본 처리 포함)
        df = pd.read_clipboard(sep=sep)
    except Exception as e:
        print(f"클립보드를 읽을 수 없습니다. 데이터가 복사되었는지 확인하세요.\n에러: {e}")
        return pd.DataFrame()

    # 2. 여백 복사로 인해 생성된 '완전히 빈' 행/열 제거
    df = df.dropna(axis=0, how='all')
    df = df.dropna(axis=1, how='all')

    # 3. 'Unnamed' 컬럼 안전 처리 (가장 중요한 부분)
    # 무작정 지우지 않고, 데이터가 '진짜로 비어있는지' 확인 후 제거합니다.
    unnamed_cols = [col for col in df.columns if str(col).startswith('Unnamed')]
    cols_to_drop = []
    
    for col in unnamed_cols:
        # 해당 열이 모두 NaN이거나 빈 문자열('')인 경우에만 삭제 후보로 등록
        if df[col].isna().all() or (df[col] == '').all():
            cols_to_drop.append(col)
            
    df = df.drop(columns=cols_to_drop)

    # 4. '.1', '.2' 등 중복 컬럼명 접미사 제거 (옵션)
    if force_rename_duplicates:
        new_cols = []
        for col in df.columns:
            # 정규식: 끝이 '.숫자'로 끝나는 패턴 찾아서 빈 문자열로 치환
            clean_col = re.sub(r'\.\d+$', '', str(col))
            new_cols.append(clean_col)
        df.columns = new_cols

    # 5. 인덱스 리셋 (복사 과정에서 엉킨 인덱스를 0부터 깔끔하게 재정렬)
    df = df.reset_index(drop=True)

    return df

# --- 실행 예시 ---
# 엑셀/웹 그리드에서 복사했을 때 (기본값 탭 구분자)
# df_loaded = read_clipboard_robustly()

# 메모장에서 쉼표(,)로 구분된 CSV 텍스트를 복사했을 때
# df_loaded = read_clipboard_robustly(sep=',')
