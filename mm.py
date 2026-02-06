import logging
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional, Final

# 로거 설정 (모듈 레벨에서 설정)
logger = logging.getLogger(__name__)

# 상수 정의 (하드코딩 제거)
KEYWORD_PRDT: Final = "PRDT"
CODE_N87: Final = "N87"
CODE_N86: Final = "N86"
ANHYDROUS_KEY: Final = "211"  # 무수물 판단 키

def _get_model_config(
    semi_product_code: str, 
    is_anhydrous: bool
) -> Tuple[bool, Dict[str, Path]]:
    """
    반제품 코드와 무수물 여부에 따라 적절한 모델 설정(경로)을 반환합니다.
    
    Returns:
        (성공여부, 모델경로Dict)
    """
    try:
        if CODE_N87 in semi_product_code:
            logger.info(f"모델 선택: {CODE_N87} 전용 모델을 사용합니다.")
            return check_model_directory("config_n87") # 예시 인자
            
        elif is_anhydrous and CODE_N86 in semi_product_code:
            logger.info(f"모델 선택: {CODE_N86} 무수물({ANHYDROUS_KEY}) 전용 모델을 사용합니다.")
            return check_model_directory("config_n86_anhydrous")
            
        else:
            logger.info("모델 선택: 기본(Default) 모델을 사용합니다.")
            return check_model_directory("config_default")
            
    except Exception as e:
        logger.error(f"모델 경로 확인 중 에러 발생: {e}")
        return False, {}

def main(
    notices_collector: "NoticeCollector", 
    execute_path: Path, 
    param_csv_path: str, 
    param_csv: pd.DataFrame
) -> Tuple[int, Optional[Path]]:
    """
    메인 실행 함수: 데이터 전처리 -> 모델 예측 -> 결과 저장을 수행합니다.
    """
    logger.info("=" * 50)
    logger.info("메인 프로세스를 시작합니다.")
    
    # 1. 결과 파일 저장 경로 확보
    try:
        model_output_path = extract_xlsx_output_path(param_csv, keyword=KEYWORD_PRDT)
        logger.debug(f"결과 저장 경로: {model_output_path}")
    except Exception as e:
        logger.error(f"결과 경로 생성 실패: {e}")
        notices_collector.fail(f"System Error: {e}")
        return 1, None

    # 2. [Validation] 파라미터 고유성 검증
    # 조건: 특정 컬럼들의 조합이 유니크해야 함 (예시)
    if not check_unique_all(param_csv):
        msg = "입력 파라미터 검증 실패: 요청 데이터가 유일하지 않습니다."
        logger.warning(msg)
        notices_collector.fail(msg)
        return 2, None

    # 3. [Execution] 폴더 프로세싱 (외부 출력 캡처)
    logger.info("폴더 데이터 스캔 및 처리를 시작합니다.")
    with capture_prints() as cap:
        processor = FolderProcessor(execute_path)
        is_folder_ok, folder_dict = processor.run(param_csv_path, param_csv)

    if not is_folder_ok:
        msg = f"폴더 처리 중 오류 발생. Output: {cap.output}"
        logger.error(msg)
        notices_collector.fail(msg)
        return 3, None

    # 4. [Pre-processing] 메타 데이터 추출 및 환경 설정
    try:
        # 데이터프레임의 첫 번째 행에서 핵심 정보 추출
        first_row = param_csv.iloc[0]
        semi_product_code = first_row["반제품코드"]
        
        # 버전 정보 추출 (단일 버전인 경우에만 사용)
        version = None
        if param_csv["Version"].nunique() == 1:
            version = first_row["Version"]
        
        # 무수물(Anhydrous) 여부 판단
        is_anhydrous = ANHYDROUS_KEY in folder_dict.keys()
        
        logger.debug(f"분석 환경 정보 - Code: {semi_product_code}, Version: {version}, Anhydrous: {is_anhydrous}")

        # 파일 로드 및 데이터 병합
        merge_all = MergeAll(folder_dict, version)
        end2end_data = merge_all.load_files(is_anhydrous)
        logger.info(f"데이터 로드 완료. Shape: {end2end_data.shape}")

    except Exception as e:
        logger.error(f"데이터 병합 과정 중 예외 발생: {e}", exc_info=True)
        notices_collector.fail(f"Data Load Error: {e}")
        return 4, None

    # 5. [Model Selection] 모델 경로 및 유효성 확인
    is_model_ok, model_path_dict = _get_model_config(semi_product_code, is_anhydrous)

    if not is_model_ok:
        msg = "적합한 모델 경로를 찾을 수 없거나 모델 파일이 누락되었습니다."
        logger.error(msg)
        notices_collector.fail(msg)
        return 5, None

    # 6. [Prediction] 예측 수행 및 결과 내보내기
    try:
        logger.info("모델 예측을 시작합니다...")
        pred_df = predict_all_targets(end2end_data, model_path_dict)
        
        logger.info(f"결과 엑셀 저장을 시도합니다. Path: {model_output_path}")
        export_path = export_predictions_to_excel(pred_df, model_output_path)
        
        logger.info("메인 프로세스가 성공적으로 완료되었습니다.")
        logger.info("=" * 50)
        return 0, export_path

    except Exception as e:
        logger.error(f"예측 또는 엑셀 저장 중 치명적 오류: {e}", exc_info=True)
        notices_collector.fail(f"Prediction/Export Error: {e}")
        return 6, None
