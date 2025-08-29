# run.py
from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


# ========= 외부 구현 연결부(당신 프로젝트에 맞게 import 교체) =========
# from yourpkg.processing import FolderProcessor, process
# from yourpkg.merge import mergeAll
# from yourpkg.modeling import CatBoostRunner, predict_all_target
# from yourpkg.export import export_prediction

# ---- 데모/독립 실행용 더미(실제 프로젝트에선 위 import 사용) ----
class FolderProcessor:
    def __init__(self) -> None: ...
class process:
    @staticmethod
    def run(filepath: Path | str, param_csv: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        return True, {"example": "dict"}
class mergeAll:
    def __init__(self, folder_dict: Dict[str, Any], version: str) -> None:
        self.folder_dict = folder_dict; self.version = version
    def load_files(self) -> pd.DataFrame:
        return pd.DataFrame({"x": [1, 2], "y": [3, 4]})
class CatBoostRunner:
    def load_model(self, *args: Any, **kwargs: Any) -> None: ...
    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series([0.1] * len(X), name="prediction")
def predict_all_target(model: CatBoostRunner, data: pd.DataFrame) -> pd.DataFrame:
    s = model.predict(data); return s.to_frame()
def export_prediction(pred_df: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True); pred_df.to_csv(out_path, index=False); return out_path
# ------------------------------------------------------------------


# ===================== Notice(예외 기록) =====================

@dataclass
class NoticeEntry:
    """단일 예외 기록 항목."""
    code: str
    message: str
    exception_type: str
    traceback_str: str
    context: Optional[Dict[str, Any]] = None


@dataclass
class NoticeCollector:
    """예외만 수집하여 엑셀로 내보낸다."""
    items: List[NoticeEntry] = field(default_factory=list)

    def add_exception(self, code: str, message: str, exc: BaseException, context: Optional[Dict[str, Any]] = None) -> None:
        """예외를 수집한다."""
        # 핵심 원리: 예외 타입/트레이스백을 구조화해 저장
        self.items.append(
            NoticeEntry(
                code=code,
                message=message,
                exception_type=exc.__class__.__name__,
                traceback_str="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                context=context,
            )
        )

    def has_items(self) -> bool:
        """수집된 예외가 있는지."""
        return len(self.items) > 0

    def to_dataframe(self) -> pd.DataFrame:
        """엑셀 저장용 DataFrame 생성."""
        rows = []
        for it in self.items:
            rows.append(
                {
                    "code": it.code,
                    "message": it.message,
                    "exception_type": it.exception_type,
                    "traceback": it.traceback_str,
                    "context_json": json.dumps(it.context, ensure_ascii=False) if it.context else None,
                }
            )
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["code", "message", "exception_type", "traceback", "context_json"])

    def write_excel(self, path: Path) -> Optional[Path]:
        """수집된 예외가 있을 때만 notice 엑셀을 쓴다.

        Args:
            path: 저장 경로 (예: notice/notice.xlsx)

        Returns:
            실제 저장된 경로 또는 None(예외 미존재/저장 스킵).
        """
        if not self.has_items():
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            self.to_dataframe().to_excel(w, sheet_name="exceptions", index=False)
        return path


# ===================== CLI/입력 =====================

def parsing_filepath() -> Path:
    """CLI 인자로 전달된 파라미터 CSV 경로를 반환한다.

    Returns:
        CSV 파일 경로.
    """
    # 핵심 원리: argparse로 --params 경로 수신
    parser = argparse.ArgumentParser(description="Run end-to-end pipeline.")
    parser.add_argument("--params", type=str, required=True, help="Path to parameter CSV file.")
    args = parser.parse_args()
    return Path(args.params)


# ===================== 유틸/검증 =====================

def read_params_csv(filepath: Path | str, encoding_candidates: Iterable[str] = ("utf-8", "cp949", "euc-kr")) -> pd.DataFrame:
    """파라미터 CSV를 안전하게 읽는다.

    Args:
        filepath: CSV 경로.
        encoding_candidates: 인코딩 후보.

    Returns:
        로드된 DataFrame.

    Raises:
        FileNotFoundError, UnicodeDecodeError, pd.errors.EmptyDataError
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    last: Optional[Exception] = None
    for enc in encoding_candidates:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last = e
            continue
    if last:
        raise last
    return pd.read_csv(path)


def validate_unique_or_raise(df: pd.DataFrame, subset: Optional[Iterable[str]] = None) -> None:
    """유니크 조건을 만족하지 않으면 예외를 발생시킨다.

    Args:
        df: 검사 대상 DataFrame.
        subset: 유니크 기준 컬럼 목록(None이면 전체 행 기준).

    Raises:
        ValueError: 중복 행이 존재할 때.
    """
    # 핵심 원리: duplicated 마스크로 중복 검출
    dup_mask = df.duplicated(subset=list(subset) if subset else None, keep=False)
    if dup_mask.any():
        dup_count = int(dup_mask.sum())
        raise ValueError(f"Parameter CSV has duplicated rows: {dup_count}")


# ===================== 파이프라인 단계 =====================

def run_processing_pipeline(param_csv: pd.DataFrame, filepath: Path | str, unique_subset: Optional[Iterable[str]] = None) -> Tuple[bool, Dict[str, Any]]:
    """process.run 실행 전 유니크성 검증 후 파이프라인을 진행한다.

    Returns:
        (is_compared, folder_dict)
    """
    # 핵심 원리: 유니크 위반은 예외로 승격 → 상위에서 일괄 기록
    validate_unique_or_raise(param_csv, unique_subset)
    processor = FolderProcessor()  # 실제 구현에서 필요시 사용
    is_compared, folder_dict = process.run(filepath, param_csv)
    return is_compared, folder_dict


def run_merge_and_load(folder_dict: Dict[str, Any], version: str) -> pd.DataFrame:
    """mergeAll로 병합/로딩."""
    merger = mergeAll(folder_dict, version)
    return merger.load_files()


def load_model_safely(*load_args: Any, **load_kwargs: Any) -> CatBoostRunner:
    """CatBoost 모델 로딩."""
    model = CatBoostRunner()
    model.load_model(*load_args, **load_kwargs)
    return model


def run_inference_and_export(model: CatBoostRunner, data: pd.DataFrame, export_path: Path) -> Path:
    """예측 → 저장."""
    pred_df = predict_all_target(model, data)
    return export_prediction(pred_df, export_path)


# ===================== 메인/에러 수거 & 저장 =====================

NOTICE_XLSX: Path = Path("notice") / "notice.xlsx"

def main() -> int:
    """메인 진입점.

    Returns:
        종료 코드(0=성공, 그 외=실패).
    """
    # 입력 경로 확보
    params_path: Path = parsing_filepath()

    # CSV 로드
    param_csv: pd.DataFrame = read_params_csv(params_path)

    # 파이프라인
    is_compared, folder_dict = run_processing_pipeline(
        param_csv=param_csv,
        filepath=params_path,
        unique_subset=None,  # 필요 시 ("LOT","DATE") 등으로 지정
    )
    if not is_compared:
        # 비교 실패 자체는 “문제 상태”로 간주 → 예외로 승격하여 notice에 남긴다.
        raise RuntimeError("Comparison step returned False.")

    # 병합/로딩
    version = "v1"
    end2end_data: pd.DataFrame = run_merge_and_load(folder_dict, version)

    # 모델 로딩
    model = load_model_safely()  # 예: load_model(model_path="model.cbm")

    # 추론/저장
    out_path = Path("output") / "predictions.csv"
    run_inference_and_export(model, end2end_data, out_path)

    return 0


def safe_main() -> int:
    """예외를 수거하여 notice.xlsx에 기록하는 래퍼."""
    notices = NoticeCollector()
    exit_code = 99
    try:
        exit_code = main()
        return exit_code
    except (pd.errors.EmptyDataError, FileNotFoundError, UnicodeDecodeError, ValueError, RuntimeError) as e:
        # 핵심 원리: 알려진 예외는 코드별로 구분
        code_map = {
            pd.errors.EmptyDataError: "CSV_EMPTY",
            FileNotFoundError: "FILE_NOT_FOUND",
            UnicodeDecodeError: "ENCODING_FAIL",
            ValueError: "VALIDATION_FAIL",
            RuntimeError: "PIPELINE_FAIL",
        }
        code = code_map.get(e.__class__, "KNOWN_ERROR")
        notices.add_exception(code=code, message=str(e), exc=e, context=None)
        exit_map = {
            "CSV_EMPTY": 10,
            "FILE_NOT_FOUND": 11,
            "ENCODING_FAIL": 12,
            "VALIDATION_FAIL": 13,
            "PIPELINE_FAIL": 14,
        }
        exit_code = exit_map.get(code, 98)
        return exit_code
    except Exception as e:
        # 핵심 원리: 알 수 없는 예외는 UNHANDLED로 기록
        notices.add_exception(code="UNHANDLED", message=str(e), exc=e, context=None)
        exit_code = 99
        return exit_code
    finally:
        # 예외가 하나라도 있으면 notice 엑셀 저장
        try:
            notices.write_excel(NOTICE_XLSX)
        except Exception:
            # 저장 실패 시에는 더 이상 할 일 없음(루프 방지)
            pass


if __name__ == "__main__":
    # Windows PyInstaller & multiprocessing 대비
    if sys.platform.startswith("win"):
        try:
            from multiprocessing import freeze_support
            freeze_support()
        except Exception:
            pass
    sys.exit(safe_main())



# run.py
from __future__ import annotations

import json
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from loguru import logger

# ===== 실제 프로젝트 구현과 연결할 import =====
# 아래 주석을 당신 프로젝트 구조에 맞게 해제/수정하세요.
# from yourpkg.io import parsing_filepath
# from yourpkg.processing import FolderProcessor, process
# from yourpkg.merge import mergeAll
# from yourpkg.modeling import CatBoostRunner, predict_all_target
# from yourpkg.export import export_prediction

# --------- 최소 더미 함수 (실 프로젝트에선 위 import로 대체) ----------
def parsing_filepath() -> Path:
    """외부 입력/GUI/CLI 등에서 파라미터 CSV 경로를 획득한다(실 구현으로 교체)."""
    return Path("params.csv")

class FolderProcessor:
    def __init__(self) -> None:
        pass

class process:
    @staticmethod
    def run(filepath: Path | str, param_csv: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        return True, {"example": "dict"}

class mergeAll:
    def __init__(self, folder_dict: Dict[str, Any], version: str) -> None:
        self.folder_dict = folder_dict
        self.version = version
    def load_files(self) -> pd.DataFrame:
        return pd.DataFrame({"x": [1,2], "y":[3,4]})

class CatBoostRunner:
    def __init__(self) -> None:
        self._loaded = False
    def load_model(self, *args: Any, **kwargs: Any) -> None:
        self._loaded = True
    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series([0.1]*len(X), name="prediction")

def predict_all_target(model: CatBoostRunner, data: pd.DataFrame) -> pd.DataFrame:
    preds = model.predict(data)
    return preds.to_frame() if isinstance(preds, pd.Series) else pd.DataFrame({"prediction": preds})

def export_prediction(pred_df: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_path, index=False)
    return out_path
# -------------------------------------------------------------------


# ===== Notice 기록 시스템 =====

@dataclass
class NoticeItem:
    """Notice 단일 항목."""
    level: str
    code: str
    message: str
    context: Optional[Dict[str, Any]] = None

@dataclass
class NoticeBuffer:
    """엑셀로 내보낼 Notice를 버퍼링."""
    items: List[NoticeItem] = field(default_factory=list)

    def add(self, level: str, code: str, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        # 핵심 원리: 실행 중 알림/에러를 구조화하여 수집
        self.items.append(NoticeItem(level=level, code=code, message=message, context=context))

    def to_dataframe(self) -> pd.DataFrame:
        # 핵심 원리: context는 JSON 문자열로 직렬화
        rows = []
        for it in self.items:
            rows.append({
                "level": it.level,
                "code": it.code,
                "message": it.message,
                "context_json": json.dumps(it.context, ensure_ascii=False) if it.context else None,
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["level","code","message","context_json"])

    def write_excel(self, path: Path, extra_sheets: Optional[Dict[str, pd.DataFrame]] = None) -> Path:
        """Notice를 지정 엑셀 경로로 저장.

        Args:
            path: 저장 경로 (예: notice/notice.xlsx).
            extra_sheets: 추가로 함께 저장할 시트들(예: 중복행 등).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            self.to_dataframe().to_excel(writer, sheet_name="notices", index=False)
            if extra_sheets:
                for name, df in extra_sheets.items():
                    df.to_excel(writer, sheet_name=name[:31] or "extra", index=False)
        return path


# ===== 로깅 설정 =====

def configure_logging(level: str = "INFO") -> None:
    """loguru 로깅 설정(개발용 콘솔 출력)."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} {level} [{name}:{line}] {message}",
        backtrace=False,
        diagnose=False,
    )


# ===== 유틸/검증 =====

def read_params_csv(
    filepath: Path | str,
    notice: NoticeBuffer,
    encoding_candidates: Iterable[str] = ("utf-8", "cp949", "euc-kr"),
) -> pd.DataFrame:
    """파라미터 CSV를 안전하게 읽는다.

    Args:
        filepath: CSV 경로.
        notice: NoticeBuffer 인스턴스.
        encoding_candidates: 인코딩 후보.

    Returns:
        로드된 DataFrame.

    Raises:
        FileNotFoundError, UnicodeDecodeError, pd.errors.EmptyDataError
    """
    path = Path(filepath)
    if not path.exists():
        # 경로 오류 notice
        notice.add("ERROR", "CSV_NOT_FOUND", "파라미터 CSV를 찾을 수 없습니다.", {"path": str(path)})
        raise FileNotFoundError(f"CSV not found: {path}")

    last_exc: Optional[Exception] = None
    for enc in encoding_candidates:
        try:
            df = pd.read_csv(path, encoding=enc)
            logger.info(f"CSV loaded with encoding='{enc}': {path.name}")
            notice.add("INFO", "CSV_LOADED", "CSV 로드 성공", {"path": str(path), "encoding": enc, "shape": list(df.shape)})
            return df
        except UnicodeDecodeError as e:
            last_exc = e
            notice.add("WARNING", "CSV_DECODE_FAIL", "인코딩 시도 실패", {"path": str(path), "encoding": enc})
            continue
    # 모든 인코딩 실패
    notice.add("ERROR", "CSV_DECODE_ALL_FAIL", "모든 인코딩 후보에서 CSV 해독에 실패했습니다.", {"path": str(path), "encodings": list(encoding_candidates)})
    if last_exc:
        raise last_exc
    # 방어적 코드
    return pd.read_csv(path)


def validate_unique(
    df: pd.DataFrame,
    notice: NoticeBuffer,
    subset: Optional[Iterable[str]] = None
) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
    """유니크성 검사.

    Args:
        df: 대상 DataFrame.
        notice: NoticeBuffer.
        subset: 유니크 기준 컬럼 목록(None이면 전체 행 기준).

    Returns:
        (유니크 판정 시리즈, 중복행 DF or None)
    """
    dup_mask = df.duplicated(subset=list(subset) if subset else None, keep=False)
    ok = ~dup_mask
    if ok.all():
        notice.add("INFO", "PARAM_UNIQUE_OK", "파라미터 유니크성 검사를 통과했습니다.", {"subset": list(subset) if subset else "ALL"})
        return ok, None
    # 중복행 수집
    dup_df = df[dup_mask].copy()
    notice.add(
        "ERROR",
        "PARAM_DUPLICATED",
        f"파라미터에 중복 행이 {len(dup_df)}건 존재합니다.",
        {"subset": list(subset) if subset else "ALL", "dup_count": int(len(dup_df))}
    )
    return ok, dup_df


# ===== 파이프라인 단계 =====

def run_processing_pipeline(
    param_csv: pd.DataFrame,
    filepath: Path | str,
    notice: NoticeBuffer,
    unique_subset: Optional[Iterable[str]] = None,
) -> Tuple[bool, Dict[str, Any], Optional[pd.DataFrame]]:
    """폴더 처리 파이프라인.

    Returns:
        (is_compared, folder_dict, dup_df)
    """
    # 유니크 검증
    check_unique, dup_df = validate_unique(param_csv, notice, unique_subset)
    if not check_unique.all():
        logger.error("Parameter CSV duplication detected.")
        return False, {}, dup_df

    # 폴더 처리
    processor = FolderProcessor()
    logger.info("FolderProcessor initialized.")
    is_compared, folder_dict = process.run(filepath, param_csv)
    notice.add(
        "INFO",
        "PROCESS_RUN_DONE",
        "process.run 완료",
        {"is_compared": bool(is_compared), "folder_keys": list(folder_dict.keys()) if folder_dict else []}
    )
    return is_compared, folder_dict, None


def run_merge_and_load(folder_dict: Dict[str, Any], version: str, notice: NoticeBuffer) -> pd.DataFrame:
    """mergeAll로 병합/로딩."""
    merger = mergeAll(folder_dict, version)
    end2end = merger.load_files()
    notice.add("INFO", "MERGE_COMPLETED", "데이터 병합 완료", {"shape": list(end2end.shape), "version": version})
    return end2end


def load_model_safely(cat_model: CatBoostRunner, notice: NoticeBuffer, *load_args: Any, **load_kwargs: Any) -> CatBoostRunner:
    """CatBoost 모델 로딩."""
    cat_model.load_model(*load_args, **load_kwargs)
    notice.add("INFO", "MODEL_LOADED", "CatBoost 모델 로드 완료", {})
    return cat_model


def run_inference_and_export(
    model: CatBoostRunner,
    data: pd.DataFrame,
    export_path: Path,
    notice: NoticeBuffer,
) -> Path:
    """예측 → 저장."""
    pred_df = predict_all_target(model, data)
    saved = export_prediction(pred_df, export_path)
    notice.add("INFO", "PRED_EXPORTED", "예측 결과 내보내기 완료", {"rows": int(len(pred_df)), "out_path": str(saved)})
    return saved


# ===== 메인 엔트리 =====

NOTICE_XLSX: Path = Path("notice") / "notice.xlsx"  # 반드시 적용 경로

def main() -> Tuple[int, NoticeBuffer, Dict[str, pd.DataFrame]]:
    """메인 로직.

    Returns:
        (exit_code, notice_buffer, extra_sheets)
    """
    configure_logging(level="INFO")
    logger.info("=== Application start ===")

    notice = NoticeBuffer()
    extra_sheets: Dict[str, pd.DataFrame] = {}

    # 입력 경로
    filepath: Path = parsing_filepath()
    notice.add("INFO", "PARAM_PATH", "파라미터 CSV 경로 확보", {"path": str(filepath)})

    # CSV 로드
    param_csv: pd.DataFrame = read_params_csv(filepath, notice)

    # 파이프라인
    is_compared, folder_dict, dup_df = run_processing_pipeline(
        param_csv=param_csv,
        filepath=filepath,
        notice=notice,
        unique_subset=None,  # 예: ("LOT", "DATE") 등 실제 키로 지정 가능
    )
    if dup_df is not None and not dup_df.empty:
        extra_sheets["duplicated_rows"] = dup_df

    if not is_compared:
        notice.add("ERROR", "COMPARE_FAILED", "비교 단계 실패 또는 이슈 발견으로 중단.", {})
        return 2, notice, extra_sheets

    # 병합/로딩
    version = "v1"
    end2end_data: pd.DataFrame = run_merge_and_load(folder_dict, version, notice)

    # 모델 로딩
    cat_model = CatBoostRunner()
    load_model_safely(cat_model, notice)  # 예: model_path="model.cbm"

    # 추론/저장
    out_path = Path("output") / "predictions.csv"
    run_inference_and_export(cat_model, end2end_data, out_path, notice)

    notice.add("INFO", "PIPELINE_DONE", "전체 파이프라인 완료", {})
    logger.info("=== Application finished successfully ===")
    return 0, notice, extra_sheets


def safe_main() -> int:
    """예외 안전 래퍼: 항상 notice.xlsx를 기록한다."""
    exit_code: int = 99
    notice: NoticeBuffer = NoticeBuffer()
    extra_sheets: Dict[str, pd.DataFrame] = {}
    try:
        exit_code, notice, extra_sheets = main()
        return exit_code
    except pd.errors.EmptyDataError as e:
        logger.exception("Empty CSV.")
        notice.add("ERROR", "CSV_EMPTY", "CSV 파일이 비어 있습니다.", {"exception": str(e)})
        exit_code = 10
        return exit_code
    except FileNotFoundError as e:
        logger.exception("File not found.")
        notice.add("ERROR", "FILE_NOT_FOUND", "필요한 파일을 찾을 수 없습니다.", {"exception": str(e)})
        exit_code = 11
        return exit_code
    except UnicodeDecodeError as e:
        logger.exception("Encoding failed.")
        notice.add("ERROR", "ENCODING_FAIL", "CSV 인코딩에 실패했습니다.", {"exception": str(e)})
        exit_code = 12
        return exit_code
    except Exception as e:
        logger.error(f"Unhandled exception: {e}\n{traceback.format_exc()}")
        notice.add("ERROR", "UNHANDLED", "처리되지 않은 예외가 발생했습니다.", {"exception": str(e)})
        exit_code = 99
        return exit_code
    finally:
        # >>> 반드시 notice 엑셀을 기록 <<<
        try:
            path = NOTICE_XLSX
            saved = notice.write_excel(path, extra_sheets=extra_sheets or None)
            logger.info(f"Notice written to: {saved}")
        except Exception as e:
            # notice 저장 자체 실패는 콘솔 로그로만 남김(루프 방지)
            logger.error(f"Failed to write notice Excel: {e}")


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        try:
            from multiprocessing import freeze_support
            freeze_support()
        except Exception:
            pass
    sys.exit(safe_main())



from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

def _ensure_xlsx_path(path: str | Path) -> Path:
    p = Path(path)
    if p.suffix.lower() != ".xlsx":
        raise ValueError(f"엑셀 경로는 .xlsx 여야 합니다: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def _prepare_X_for_model(end2end_data: pd.DataFrame, model) -> pd.DataFrame:
    """
    end2end_data의 컬럼을 model.X_test.columns 순서로 정렬.
    - end2end_data가 필요한 컬럼을 모두 포함하는지 확인
    - 추가 컬럼은 무시(사용 안함)
    """
    if not hasattr(model, "X_test") or not hasattr(model.X_test, "columns"):
        raise AttributeError("model.X_test.columns 를 찾을 수 없습니다.")
    req_cols = list(model.X_test.columns)

    # 누락 컬럼 검사
    missing = [c for c in req_cols if c not in end2end_data.columns]
    if missing:
        raise ValueError(f"입력 데이터에 누락된 컬럼: {missing}")

    # 모델이 요구하는 순서로 재정렬
    X_ordered = end2end_data[req_cols].copy()
    # (선택) 타입/NaN 전처리가 필요하면 여기서 추가
    return X_ordered

def predict_cat_to_seven(cat_model, end2end_data: pd.DataFrame) -> pd.DataFrame:
    """
    - cat_model 기준으로 컬럼 순서 확인/정렬
    - cat_model 예측 1개를 7개 컬럼으로 복제
    - index는 end2end_data의 index를 그대로 사용
    - 항상 (n_rows x 7) DataFrame 반환
    """
    X = _prepare_X_for_model(end2end_data, cat_model)
    y = np.asarray(cat_model.predict(X)).reshape(-1)
    if len(y) != len(X.index):
        raise ValueError("예측 길이와 입력 행수가 일치하지 않습니다.")

    out = pd.DataFrame(
        {f"model_{i}": y for i in range(1, 8)},
        index=X.index  # 입력 인덱스 유지
    )
    return out
https://github.com/yc-jang/pv_repo/pull/10
def export_predictions_to_excel(df: pd.DataFrame, model_output_path: str | Path) -> Path:
    """
    결과 DataFrame을 지정된 엑셀 경로의 Sheet1에 저장(덮어쓰기).
    """
    out = _ensure_xlsx_path(model_output_path)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Sheet1", index=True)
    return out

# ========= 사용 예 =========
# pred_df = predict_cat_to_seven(cat_model, end2end_data)
# final_path = export_predictions_to_excel(pred_df, "C:/result/model_output.xlsx")
# print("저장 완료:", final_path)


from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

def _ensure_xlsx_path(path: str | Path) -> Path:
    p = Path(path)
    if p.suffix.lower() != ".xlsx":
        raise ValueError(f"엑셀 경로는 .xlsx 여야 합니다: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)  # 상위 폴더 자동 생성
    return p

def export_df_to_excel(df: pd.DataFrame, output_path: str | Path, sheet_name: str = "predictions") -> Path:
    """DataFrame을 지정된 .xlsx 경로에 저장(폴더 자동 생성)."""
    out = _ensure_xlsx_path(output_path)
    # 항상 덮어쓰기; 필요 시 mode/if_sheet_exists 옵션 확장 가능
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=True, sheet_name=sheet_name)
    return out

def build_model_list(cat_model, n_models: int = 7, model_names: list[str] | None = None):
    """
    cat_model 하나를 n_models개로 '복제 사용'하는 리스트 구성.
    실제 복제(copy)가 필요 없다면 동일 객체 참조로 충분.
    """
    if model_names is None:
        model_names = [f"model_{i+1}" for i in range(n_models)]
    if len(model_names) != n_models:
        raise ValueError("model_names 길이가 n_models와 다릅니다.")
    models = [(name, cat_model) for name in model_names]  # 동일 모델 재사용
    return models

def predict_models_to_df(models: list[tuple[str, object]], X: pd.DataFrame) -> pd.DataFrame:
    """
    여러 모델의 예측을 하나의 DataFrame으로 반환.
    - X.index를 그대로 유지
    - 단일 모델이어도 DataFrame 형식 보장 (1열)
    """
    preds = {}
    for name, model in models:
        y = model.predict(X)  # CatBoost의 predict 결과
        y = np.asarray(y).reshape(-1)  # 1차원 보장
        if len(y) != len(X.index):
            raise ValueError(f"{name} 예측 길이({len(y)})와 X 행수({len(X)}) 불일치")
        preds[name] = y
    out_df = pd.DataFrame(preds, index=X.index)
    return out_df

def run_and_export(cat_model, X: pd.DataFrame, output_xlsx: str | Path,
                   n_models: int = 7, model_names: list[str] | None = None,
                   sheet_name: str = "predictions") -> Path:
    """
    - cat_model을 n_models개로 복제 사용하여 예측
    - X.index 유지 + 항상 DataFrame 반환
    - 지정된 엑셀(output_xlsx)에 저장 보장(폴더 자동 생성)
    - 최종 저장 경로 반환
    """
    models = build_model_list(cat_model, n_models=n_models, model_names=model_names)
    pred_df = predict_models_to_df(models, X)
    return export_df_to_excel(pred_df, output_xlsx, sheet_name=sheet_name)
