import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from loguru import logger

# ==============================================================================
# [USER CONFIGURATION] 사용자 설정 영역
# 파일 경로를 변경하거나 기본 설정을 수정하려면 아래 변수들을 변경하세요.
# ==============================================================================

# 기본 데이터 파일 경로 (터미널 인자가 없을 때 사용됨)
DEFAULT_DATA_PATH = "./data/merged_data.csv"

# 학습할 Target 컬럼 목록
TARGET_COLUMNS = ["Target_A", "Target_B"] 

# 전체 Target 컬럼 목록 (Config용)
ALL_TARGET_COLUMNS = ["Target_A", "Target_B", "Target_C"]

# 학습에서 제외할 불필요 컬럼
DISCUSSED_DROP_COLUMNS = ["Unnecessary_Col1", "Unnecessary_Col2"]

# 인덱스로 사용할 컬럼
INDEX_COLUMNS = ["JOB_CODE", "TIMESTAMP"]

# ==============================================================================

# (가상의 의존성 함수들 - 실제 환경에서는 import 해서 사용)
# from my_module import load_csv_data, train_evaluate_save, CatConfig
def load_csv_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

class CatConfig:
    def __init__(self, target, all_target_columns, use_optuna, n_trial):
        pass

def train_evaluate_save(data, cfg):
    return {"score": 0.99, "model_path": "model.pkl"}


def run_training(
    data_path: Path,
    target_columns: List[str],
    all_target_columns: List[str],
    use_optuna: bool = False,
    n_trial: int = 100,
) -> Dict[str, Any]:
    """
    병합된 단일 데이터로 여러 TARGET을 순회하며 학습을 수행합니다.
    """
    results: Dict[str, Any] = {}

    # --- (0) 파일 존재 여부 확인 (Pre-check) ---
    if not data_path.exists():
        logger.critical(f"Data file not found: {data_path}")
        logger.error("Please check the file path in [USER CONFIGURATION] or command line arguments.")
        return {"error": "File not found"}

    try:
        # --- (1) 데이터 로드 & 공통 전처리 ---
        logger.info(f"Loading training data from: {data_path}")
        data = load_csv_data(data_path)

        logger.info("Dropping discussed columns...")
        data = data.drop(columns=DISCUSSED_DROP_COLUMNS, errors="ignore")

        # 인덱스 설정은 반복문 밖에서 1회 수행 (성능 최적화)
        if all(col in data.columns for col in INDEX_COLUMNS):
            logger.info(f"Setting index to {INDEX_COLUMNS}")
            data = data.set_index(INDEX_COLUMNS)
        else:
            logger.warning(f"Index columns {INDEX_COLUMNS} not found. Skipping set_index.")

    except Exception as e:
        logger.exception("Failed during data loading or preprocessing phase.")
        return {"status": "failed", "error": str(e)}

    # --- (2) TARGET별 학습 순회 ---
    logger.info(f"Starting training loop for {len(target_columns)} targets.")
    
    for target in target_columns:
        logger.info(f"Processing Target: {target}")

        try:
            # Config 생성
            cfg = CatConfig(
                target=target,
                all_target_columns=all_target_columns,
                use_optuna=use_optuna,
                n_trial=n_trial,
            )

            # --- Data Filtering ---
            # 해당 타겟의 결측치가 없는 데이터만 추출
            # dropna()는 새로운 객체(copy)를 반환하므로 원본 data는 안전함
            if target not in data.columns:
                logger.error(f"Target column '{target}' not found in dataset. Skipping.")
                results[target] = {"status": "skipped", "reason": "Column missing"}
                continue

            used_data = data.dropna(subset=[target])

            if used_data.empty:
                logger.warning(f"No valid rows after dropna for target={target}. Skipping.")
                results[target] = {"status": "skipped", "reason": "Empty data"}
                continue
            
            # --- Training ---
            logger.debug(f"Training with {len(used_data)} rows...")
            out = train_evaluate_save(used_data, cfg)
            
            results[target] = out
            logger.success(f"Finished training for target: {target}")

        except Exception as e:
            # 특정 타겟 실패 시 전체 중단하지 않고 로그 남기고 다음 타겟으로
            logger.exception(f"Training failed for target={target}")
            results[target] = {
                "status": "failed",
                "error": str(e),
            }

    return results


if __name__ == "__main__":
    # 터미널 실행 시 인자 파싱 (CLI 지원)
    parser = argparse.ArgumentParser(description="E2E Quality Prediction Training")
    parser.add_argument(
        "--data_path", 
        type=str, 
        default=DEFAULT_DATA_PATH, 
        help="Path to the merged CSV data file"
    )
    
    args = parser.parse_args()
    path_obj = Path(args.data_path)

    logger.info("========================================")
    logger.info("       STARTING TRAINING PROCESS        ")
    logger.info("========================================")
    
    # 실행
    final_results = run_training(
        data_path=path_obj,
        target_columns=TARGET_COLUMNS,
        all_target_columns=ALL_TARGET_COLUMNS,
        use_optuna=True,
        n_trial=50
    )

    logger.info("========================================")
    logger.info(f"Process Completed. Results: {len(final_results)} items.")
