import pandas as pd

def compare_and_report(folder_dict: dict, param_df: pd.DataFrame) -> bool:
    """
    folder_dict : {'01': 'path1', '02': 'path2', ...}
    param_df    : DataFrame with ['저장폴더','파일명']

    return : bool (True if 완벽히 일치, False otherwise)
    """
    # 1) param_df 경로 합치기
    param_paths = param_df["저장폴더"].astype(str).str.rstrip("/") + "/" + param_df["파일명"].astype(str)
    # 2) 맨 앞 "P960/" 제거
    param_paths = param_paths.str.replace(r"^P960/", "", regex=True)

    set_param = set(param_paths)
    set_folder = set(folder_dict.values())

    is_equal = set_param == set_folder

    # --- 깔끔 출력 ---
    print("="*40)
    print(f"📂 folder_dict count : {len(set_folder)}")
    print(f"📄 param_df   count : {len(set_param)}")
    print(f"✅ 완벽히 일치 여부 : {is_equal}")
    print("="*40)

    if not is_equal:
        only_in_param = sorted(set_param - set_folder)
        only_in_folder = sorted(set_folder - set_param)

        if only_in_param:
            print("⚠️ param_df 에만 있는 경로:")
            for p in only_in_param:
                print(f"   - {p}")

        if only_in_folder:
            print("⚠️ folder_dict 에만 있는 경로:")
            for f in only_in_folder:
                print(f"   - {f}")
    else:
        print("🎉 두 집합이 완벽히 일치합니다.")

    print("="*40)
    return is_equal
    
import pandas as pd
import numpy as np

def calculate_outlier_stats(file_path, lower_spec=209, upper_spec=213):
    """
    엑셀 데이터에서 관리 규격을 벗어난 통계를 계산하여 DataFrame으로 반환.
    
    Parameters:
    - file_path (str): 엑셀 파일 경로
    - lower_spec (float): 관리 규격 하한 (기본값: 209)
    - upper_spec (float): 관리 규격 상한 (기본값: 213)
    
    Returns:
    - pd.DataFrame: 컬럼별 이상치 수와 평균을 포함한 결과
    """
    try:
        # 엑셀 데이터 로드
        data = pd.read_excel(file_path)
        
        # 분석 대상 컬럼
        columns = ["PRDT_Target", "Before_Optim", "After_Optim"]
        
        # 컬럼 존재 확인
        missing_cols = [col for col in columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"다음 컬럼이 데이터에 없습니다: {missing_cols}")
        
        # 결과 저장용 리스트
        results = []
        
        # 각 컬럼별 이상치 분석
        for col in columns:
            # 규격 벗어난 데이터 필터링
            outliers = data[col][(data[col] < lower_spec) | (data[col] > upper_spec)]
            
            # 이상치 수와 평균 계산
            outlier_count = len(outliers)
            outlier_mean = outliers.mean() if outlier_count > 0 else np.nan
            
            # 결과 추가
            results.append({
                "Column": col,
                "Outlier Count": outlier_count,
                "Outlier Mean": outlier_mean
            })
        
        # DataFrame으로 변환
        result_df = pd.DataFrame(results)
        
        return result_df
    
    except FileNotFoundError:
        print(f"Error: 파일 '{file_path}'을 찾을 수 없습니다.")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# 사용 예시
file_path = "your_data.xlsx"  # 실제 파일 경로로 수정
result_df = calculate_outlier_stats(file_path, lower_spec=209, upper_spec=213)

# 결과 출력
if result_df is not None:
    print(result_df)
    
    # 엑셀로 저장
    result_df.to_excel("outlier_stats.xlsx", index=False)
    print("결과가 'outlier_stats.xlsx'로 저장되었습니다.")
