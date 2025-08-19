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
