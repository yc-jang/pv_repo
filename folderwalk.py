import os
import re
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Union
import pandas as pd

logger = logging.getLogger(__name__)

class FolderProcessor:
    # 성능 극대화를 위해 정규식을 클래스 레벨에서 한 번만 컴파일 (선행 숫자만 추출)
    NUMBER_PATTERN = re.compile(r"^(\d+)")

    def __init__(self, execute_path: Union[str, Path]):
        # 최신 경로 탐색을 위해 무조건 Path 객체로 변환하여 보유
        self.execute_path = Path(execute_path)

    def walk_folders(self, param: pd.DataFrame) -> Dict[str, Union[str, List[str]]]:
        """
        주어진 DataFrame을 순회하며 조건에 맞는 파일들을 찾고, 
        Job Code 기준으로 1:N (Dict[str, List/str]) 형태의 매핑을 반환합니다.
        """
        # 1:N 맵핑을 안전하게 수집하기 위한 임시 저장소 (기본값이 빈 리스트)
        raw_job_dict: Dict[str, List[str]] = defaultdict(list)

        for idx, trigger in param.iterrows():
            try:
                # 안전한 타입 추출 (결측치 및 타입 혼용 방어)
                project_number = str(trigger.get("pnumber", "")).strip()
                request_user = str(trigger.get("user", "")).strip()
                request_number = str(trigger.get("request_number", "")).strip()
                
                # float '0.0', int 0, str '0' 등을 모두 포용하는 안전한 버전 체크
                raw_version = trigger.get("version")
                is_version_0 = (str(raw_version).strip() in ["0", "0.0", "0.00"])
                
                io_class = str(trigger.get("io_class", "")).strip().upper()

                # [최적화 1] Guard Clause: 파일시스템 접근 전 OUT 체크로 조기 차단
                if io_class == 'OUT':
                    continue

                folder_path = self.execute_path / request_user / io_class
                
                # [최적화 2] 디렉토리 존재 유무 검증
                if not folder_path.exists() or not folder_path.is_dir():
                    logger.error(f"경로 없음 또는 폴더가 아님: {folder_path}")
                    continue

                # [최적화 3] 중첩 os.walk를 제거하고 rglob으로 전체 트리를 한 번에 순회
                for file_path in folder_path.rglob("*"):
                    try:
                        # 디렉토리 자체는 무시하고 실제 파일만 처리
                        if not file_path.is_file():
                            continue

                        # [버그 수정 1] 파일명 내부에 request_number가 부분 포함되어 있는지 확인
                        if request_number not in file_path.name:
                            continue

                        # 상대 경로를 파트별로 깔끔하게 쪼갬 (예: ('dir1', 'dir2', 'file.txt'))
                        relative_parts = file_path.relative_to(folder_path).parts

                        # [최적화 4] 버전별 폴더 제외(Exclude) 룰 안전 처리
                        if len(relative_parts) > 1:
                            # 대문자로 변환하여 오타로 인한 예외 방어
                            second_depth_folder = relative_parts[1].upper()
                            if is_version_0 and second_depth_folder == "3SIMULATION":
                                continue
                            elif not is_version_0 and second_depth_folder == "1CONDITION":
                                continue

                        # [버그 수정 2] 정규식 이중 평가 제거 및 선행 숫자 추출
                        extracted_numbers = []
                        # 파일명(마지막 part)을 제외한 상위 폴더들만 순회
                        for part in relative_parts[:-1]:
                            match = self.NUMBER_PATTERN.match(part)
                            if match:
                                extracted_numbers.append(match.group(1))

                        # Job Code 생성 (추출된 숫자가 없으면 빈 문자열을 zfill 처리함 -> "00")
                        job_code = "".join(extracted_numbers).zfill(2)

                        # 예외 룰: 전체 경로에 특정 키워드가 들어가면 코드 강제 할당
                        if "LOT_TRACKING_2" in str(file_path):
                            job_code = "211"

                        # [핵심] 찾은 파일 경로를 1:N 구조 리스트에 안전하게 누적 (절대 경로 문자열)
                        raw_job_dict[job_code].append(str(file_path))

                    except Exception as inner_e:
                        # 특정 파일에서 오류가 나도 시스템이 죽지 않고 다음 파일을 탐색
                        logger.error(f"Error processing individual file {file_path}: {inner_e}")
                        continue

            except Exception as outer_e:
                # 특정 Row 처리 중 오류가 나도 다음 Row로 계속 진행
                logger.error(f"Error processing row index {idx}: {outer_e}")
                continue

        # --- [버그 수정 3] 수동 업데이트 부분 덮어쓰기 방지 ---
        # 원본: job_folder_dict.update({"90": str(...)})
        # 개선: 기존에 수집된 "90" 데이터가 날아가지 않도록 리스트에 append
        manual_90_path = "여기에_수동_지정할_경로를_넣으세요" # 기존 str(...)에 해당하는 부분
        if manual_90_path:
            raw_job_dict["90"].append(manual_90_path)

        # --- [최종 마무리] 1:N 포맷 맞추기 ---
        # 요구사항에 따라 1개면 str, 여러 개면 list로 변환하여 최종 dict 생성
        final_job_dict: Dict[str, Union[str, List[str]]] = {}
        for code, path_list in raw_job_dict.items():
            if not path_list:
                continue
            
            # 리스트 내 동일한 파일 경로 중복 제거 (안전장치)
            unique_paths = list(dict.fromkeys(path_list))

            if len(unique_paths) == 1:
                final_job_dict[code] = unique_paths[0]
            else:
                final_job_dict[code] = unique_paths

        return final_job_dict


import logging
from pathlib import Path
import pandas as pd
from typing import Dict, Union, List

logger = logging.getLogger(__name__)

def compare_folder_param(self, folder_dict: Dict[str, Union[str, List[str]]], param_csv: pd.DataFrame) -> bool:
    """
    param_csv의 IN 조건 파일 목록과 실제 폴더 순회 결과(folder_dict)를 
    비교하여 일치 여부를 반환하고, 불일치 시 상세 원인을 로깅합니다.
    """
    try:
        # 1. 대상 데이터 필터링 (명확한 복사본 생성)
        param_in = param_csv[param_csv["입출력구분"] == "IN"].copy()
        
        # [버그 방어 1] 안전한 경로 병합 (구분자 누락 방지)
        # 폴더명 우측 슬래시 제거, 파일명 좌측 슬래시 제거 후 '/'로 안전하게 연결
        folder_col = param_in["저장폴더"].astype(str).str.replace("\\", "/", regex=False).str.rstrip("/")
        file_col = param_in["파일명"].astype(str).str.replace("\\", "/", regex=False).str.lstrip("/")
        param_in_path = folder_col + "/" + file_col

        # 2. 예외 룰 적용 (P960_PARAM 제외) - na=False로 결측치로 인한 에러 방지
        mask_valid = ~param_in_path.str.contains("P960_PARAM", case=False, na=False)
        param_in_path = param_in_path[mask_valid]

        # [버그 방어 2 & 4] 안전한 정규화 및 Set 변환
        # resolve()의 I/O 병목 대신, Path 객체의 포맷 통일 기능을 활용
        set_param = {Path(p).resolve() for p in param_in_path}

        # [버그 방어 3] 1:N 구조 (str과 list의 혼재) 완벽 대응 (Flattening)
        flat_folder_paths = []
        for paths in folder_dict.values():
            if isinstance(paths, list):
                flat_folder_paths.extend(paths)
            else:
                flat_folder_paths.append(paths)
                
        # 수집된 실제 폴더 경로들도 동일하게 포맷팅
        set_folder = {Path(p).resolve() for p in flat_folder_paths}

        # 3. 데이터 검증 및 결과 판별
        is_equal = (set_param == set_folder)

        # [최적화 핵심] 디버깅을 위한 차이점 로깅 (현업 필수 로직)
        if not is_equal:
            missing_in_folder = set_param - set_folder
            extra_in_folder = set_folder - set_param
            
            logger.warning("🚨 param_csv와 실제 폴더의 파일 구성이 일치하지 않습니다!")
            
            if missing_in_folder:
                # 너무 길어지지 않게 최대 5개까지만 샘플 출력
                sample = list(missing_in_folder)[:5]
                logger.warning(f"  -> param에는 있지만 폴더에는 없는 파일 ({len(missing_in_folder)}개): {sample} ...")
                
            if extra_in_folder:
                sample = list(extra_in_folder)[:5]
                logger.warning(f"  -> 폴더에는 있지만 param에는 없는 파일 ({len(extra_in_folder)}개): {sample} ...")

        return is_equal

    except Exception as e:
        logger.error(f"compare_folder_param 실행 중 예기치 않은 오류 발생: {e}")
        # 검증 로직 자체에서 에러가 나면 안전을 위해 무조건 불일치(False) 처리
        return False

