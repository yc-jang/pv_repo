from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple, Optional
import re
import pandas as pd
from loguru import logger


# ── 로깅 기본 설정 ───────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sink=lambda m: print(m, end=""),
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} {level} [{name}:{line}] {message}",
)


# ── 설정 데이터클래스 ────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ScanConfig:
    """스캔 및 병합 파이프라인 설정.

    Attributes:
        base_dir: 스캔 시작 루트 디렉터리.
        param_name_substr: 단위 폴더 식별 기준이 되는 파일명 부분 문자열(예: 'P960_PARAM_01').
        param_exts: 단위 파일 허용 확장자 집합(예: ('.csv',)).
        tag_meta_from_path: 경로에서 연/분기/반기 메타를 추출해 DF에 태깅 여부.
    """
    base_dir: Path
    param_name_substr: str = "P960_PARAM_01"
    param_exts: Tuple[str, ...] = (".csv",)
    tag_meta_from_path: bool = True


# ── 경로→메타 파서 (요구 규칙 반영) ─────────────────────────────────────────
_YEAR_PAT = re.compile(r"(?P<y>\d{2,4})년")
_QUARTER_PAT = re.compile(r"(?P<q>\d{2})분기")
_HALF_TOKENS = {"상반기", "하반기"}  # 현 요구는 '상반기'만 유효하게 사용

def _norm_year(two_or_four: int) -> int:
    """두 자리면 2000+ 보정, 네 자리면 그대로."""
    return two_or_four if two_or_four >= 1000 else (2000 + two_or_four)

def parse_meta_from_path(path: Path) -> dict:
    """경로 세그먼트에서 연/분기/반기 메타데이터 추출.

    Args:
        path: 폴더 경로(보통 단위 폴더).

    Returns:
        dict: {'unit_year': Optional[int], 'unit_quarter': Optional[str], 'unit_half': Optional[str]}
    """
    # 핵심 원리: 세그먼트별 정규식/토큰 매칭
    year: Optional[int] = None
    quarter: Optional[str] = None
    half: Optional[str] = None

    for part in path.parts:
        my = _YEAR_PAT.search(part)
        if my:
            year = _norm_year(int(my.group("y")))
        mq = _QUARTER_PAT.search(part)
        if mq:
            quarter = mq.group("q")
        if part in _HALF_TOKENS:
            half = part

    return {"unit_year": year, "unit_quarter": quarter, "unit_half": half}


# ── 탐색 유틸 ────────────────────────────────────────────────────────────────
def _folder_has_param_file(folder: Path, name_substr: str, exts: Tuple[str, ...]) -> bool:
    """해당 폴더에 이름 부분문자열+확장자 조건을 만족하는 파일이 있는지 검사."""
    # 핵심 원리: 파일명 substring + 확장자 동시 만족
    try:
        for fname in os.listdir(folder):
            if name_substr in fname and fname.lower().endswith(tuple(ext.lower() for ext in exts)):
                return True
    except FileNotFoundError:
        return False
    return False

def discover_unit_folders(cfg: ScanConfig) -> List[Path]:
    """os.walk로 전체를 순회하며 '단위 폴더'를 식별한다.

    단위 폴더 정의: 해당 폴더 **내부**에 cfg.param_name_substr 포함 파일이 있고,
    그 확장자가 cfg.param_exts 중 하나인 경우.

    Args:
        cfg: 스캔 설정.

    Returns:
        단위 폴더의 절대경로 리스트(정렬).
    """
    base = cfg.base_dir.resolve()
    if not base.exists():
        raise FileNotFoundError(f"base_dir not found: {base}")

    unit_folders: List[Path] = []

    # 핵심 원리: 디렉터리 단위로 검사, 히트 시 '단위 폴더'로 채택
    for root, _dirs, files in os.walk(base):
        if not files:
            continue
        folder = Path(root)
        if _folder_has_param_file(folder, cfg.param_name_substr, cfg.param_exts):
            unit_folders.append(folder.resolve())

    unit_folders = sorted(set(unit_folders), key=lambda p: p.as_posix())
    logger.info(f"Discovered unit folders: {len(unit_folders)} under {base}")
    return unit_folders


# ── 단위 병합 실행(직렬) ─────────────────────────────────────────────────────
def run_unit_merge(
    unit_folders: Iterable[Path],
    merge_unit: Callable[[Path], pd.DataFrame],
    tag_meta: bool = True,
) -> Dict[str, pd.DataFrame]:
    """단위 폴더별로 기존 merge 함수를 호출하여 결과를 수집(직렬).

    Args:
        unit_folders: 단위 폴더 경로들.
        merge_unit: 기존 단위 병합 함수(인자: 폴더 경로(Path) → 반환: 병합된 DataFrame).
        tag_meta: 경로 메타데이터(unit_year/quarter/half)를 결과 DF에 태깅 여부.

    Returns:
        {"단위폴더_절대경로": merged_df, ...}
    """
    results: Dict[str, pd.DataFrame] = {}

    for folder in unit_folders:
        try:
            # 핵심 원리: 기존 병합 함수 호출(폴더→DataFrame)
            df = merge_unit(folder).copy()
            df["unit_root"] = str(folder.resolve())  # 추적성 보강

            if tag_meta:
                meta = parse_meta_from_path(folder)
                for k, v in meta.items():
                    df[k] = v

            results[str(folder.resolve())] = df
            logger.info(f"Merged unit: {folder}")
        except Exception as e:
            # 필요 최소 예외 처리: 실패 폴더만 로깅하고 계속
            logger.error(f"Failed on {folder} | {e}")

    return results


# ── 전체 합치기 ──────────────────────────────────────────────────────────────
def concat_all(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """폴더별 병합 결과를 열 정합 후 세로 결합한다.

    Args:
        results: {"단위폴더_절대경로": merged_df, ...}

    Returns:
        전체 합본 DataFrame (열은 합집합, 누락열은 NaN).
    """
    # 핵심 원리: 열 합집합 수집 후 재인덱싱 → concat
    if not results:
        return pd.DataFrame()
    all_cols = sorted({c for df in results.values() for c in df.columns})
    aligned = [df.reindex(columns=all_cols) for df in results.values()]
    return pd.concat(aligned, axis=0, ignore_index=True)


# ── 기존 함수 어댑터(필수 교체) ─────────────────────────────────────────────
def merge_unit_adapter(unit_root: Path) -> pd.DataFrame:
    """네가 기존에 쓰던 '폴더 → 병합 DataFrame' 함수를 연결한다.

    예시:
        # 기존 함수 시그니처 예: your_merge(folder: str) -> pd.DataFrame
        return your_merge(folder=str(unit_root))

    Returns:
        해당 단위의 병합 DataFrame
    """
    # TODO: 아래 한 줄을 네 실제 병합 함수 호출로 교체하세요.
    raise NotImplementedError("merge_unit_adapter를 기존 병합 함수로 교체하세요.")


# ── 파이프라인 오케스트레이터 ───────────────────────────────────────────────
def run_pipeline(
    cfg: ScanConfig,
    merge_unit: Callable[[Path], pd.DataFrame],
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """엔드투엔드 파이프라인: 스캔→단위병합(직렬)→전체 합치기.

    Args:
        cfg: 스캔 설정.
        merge_unit: 기존 병합 함수(폴더 → DataFrame).

    Returns:
        (results_map, merged_all)
        - results_map: {"단위폴더_절대경로": 병합DF, ...}
        - merged_all: 전체 합본 DataFrame
    """
    # 1) 단위 폴더 스캔
    unit_folders = discover_unit_folders(cfg)

    # 2) 단위별 병합 실행(직렬)
    results_map = run_unit_merge(
        unit_folders=unit_folders,
        merge_unit=merge_unit,
        tag_meta=cfg.tag_meta_from_path,
    )

    # 3) 전체 concat (옵션)
    merged_all = concat_all(results_map)
    return results_map, merged_all


# ── 사용 예시(주석 해제 후 즉시 실행 가능) ──────────────────────────────────
# if __name__ == "__main__":
#     cfg = ScanConfig(
#         base_dir=Path(r"C:\P900\P002"),
#         param_name_substr="P960_PARAM_01",
#         param_exts=(".csv",),
#         tag_meta_from_path=True,
#     )
#     results_map, merged_all = run_pipeline(cfg, merge_unit=merge_unit_adapter)
#     # results_map: {"...절대경로...": df, ...}
#     # merged_all: 전체 합본 DataFrame
