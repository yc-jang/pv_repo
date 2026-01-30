import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

# ==============================================================================
# [1] USER CONFIGURATION (사용자 설정 영역)
# 아래 변수들의 값을 변경하여 실행 환경을 설정하세요.
# ==============================================================================

# 1. 학습 데이터 경로 (파일이 없으면 에러 발생)
DATA_FILE_PATH = "./data/merged_data.csv"

# 2. 학습할 Target 컬럼 목록
TARGET_COLUMNS = ["Target_A", "Target_B"]

# 3. 전체 Target 컬럼 목록 (Config 설정용)
ALL_TARGET_COLUMNS = ["Target_A", "Target_B", "Target_C"]

# 4. 학습에서 제외할 불필요 컬럼
DROP_COLUMNS = ["Unnecessary_Col1", "Unnecessary_Col2"]

# 5. 인덱스로 사용할 컬럼
INDEX_COLUMNS = ["JOB_CODE", "TIMESTAMP"]

# 6. 기타 학습 설정
USE_OPTUNA = True
N_TRIAL = 50

# ==============================================================================
# [2] LOGGING SETUP (로깅 설정)
# loguru 대신 표준 logging을 사용하여 깔끔한 로그 포맷을 설정합니다.
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("TrainingManager")

# ==============================================================================
# [3] HELPER FUNCTIONS & CLASSES (가상 모듈)
# 실제 환경에서는 import 문으로 대체하세요.
# ==============================================================================
def load_csv_data(path: Path) -> pd.DataFrame:
    """CSV 데이터를 로드합니다."""
    return pd.read_csv(path)

class CatConfig:
    def __init__(self, target, all_target_columns, use_optuna, n_trial):
        self.target = target
        # ... Config 초기화 로직

def train_evaluate_save(data: pd.DataFrame, cfg: CatConfig) -> Dict[str, Any]:
    """학습 및 평가 로직 (Placeholder)"""
    # 실제 학습 로직 수행
    return {"status": "success", "score": 0.95}

# ==============================================================================
# [4] MAIN EXECUTION LOGIC
# ==============================================================================
def run_training(
    data_path: Path,
    target_columns: List[str],
    all_target_columns: List[str],
    drop_cols: List[str],
    index_cols: List[str],
    use_optuna: bool,
    n_trial: int,
) -> Dict[str, Any]:
    """
    데이터를 1회 로드한 후, 여러 Target에 대해 순차적으로 학습을 수행합니다.
    """
    results: Dict[str, Any] = {}

    # --- 1. 파일 존재 여부 검증 ---
    if not data_path.exists():
        logger.critical(f"Data file not found at: {data_path}")
        logger.error("Please check 'DATA_FILE_PATH' in the [USER CONFIGURATION] section.")
        return {"error": "File not found"}

    try:
        # --- 2. 데이터 로드 (I/O 1회 수행) ---
        logger.info(f"Loading data from: {data_path}")
        raw_data = load_csv_data(data_path)
        logger.info(f"Data loaded successfully. Shape: {raw_data.shape}")

        # --- 3. 공통 전처리 (반복문 외부 수행) ---
        # 불필요 컬럼 제거
        if drop_cols:
            logger.info(f"Dropping columns: {drop_cols}")
            raw_data = raw_data.drop(columns=drop_cols, errors="ignore")

        # 인덱스 설정
        available_index_cols = [c for c in index_cols if c in raw_data.columns]
        if available_index_cols:
            logger.info(f"Setting index: {available_index_cols}")
            raw_data = raw_data.set_index(available_index_cols)
        else:
            logger.warning(f"Index columns {index_cols} not found. Using default index.")

    except Exception as e:
        logger.exception("Critical error during data loading/preprocessing.")
        return {"status": "critical_failure", "error": str(e)}

    # --- 4. Target별 학습 반복 (Loop) ---
    logger.info("=" * 60)
    logger.info(f"Starting training for {len(target_columns)} targets...")
    logger.info("=" * 60)

    for target in target_columns:
        logger.info(f">>> Processing Target: {target}")

        try:
            # Config 설정
            cfg = CatConfig(
                target=target,
                all_target_columns=all_target_columns,
                use_optuna=use_optuna,
                n_trial=n_trial,
            )

            # 데이터 필터링 (결측치 제거 및 복사)
            if target not in raw_data.columns:
                logger.error(f"Target column '{target}' does not exist in data. Skipping.")
                results[target] = "Skipped (Column Missing)"
                continue

            # 해당 타겟의 값이 존재하는 행만 추출 (.copy()로 원본 보호)
            used_data = raw_data.dropna(subset=[target]).copy()

            if used_data.empty:
                logger.warning(f"No valid data rows for target '{target}'. Skipping.")
                results[target] = "Skipped (Empty Data)"
                continue

            # 학습 수행
            out = train_evaluate_save(used_data, cfg)
            
            # 결과 저장
            results[target] = out
            logger.info(f"[Done] Finished training for {target}")

        except Exception as e:
            # 개별 타겟 실패 시 전체 중단 방지
            logger.error(f"Training failed for target '{target}': {e}")
            results[target] = {"status": "failed", "error": str(e)}

    logger.info("=" * 60)
    logger.info("All training processes completed.")
    
    return results

if __name__ == "__main__":
    # Path 객체 변환
    data_path_obj = Path(DATA_FILE_PATH)

    # 메인 함수 실행
    final_results = run_training(
        data_path=data_path_obj,
        target_columns=TARGET_COLUMNS,
        all_target_columns=ALL_TARGET_COLUMNS,
        drop_cols=DROP_COLUMNS,
        index_cols=INDEX_COLUMNS,
        use_optuna=USE_OPTUNA,
        n_trial=N_TRIAL
    )

    # 최종 결과 요약 출력 (선택 사항)
    # print(final_results)
