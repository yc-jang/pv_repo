from __future__ import annotations
import io
import os
from os import PathLike
from typing import Any, Iterable
import unicodedata
import re
import pandas as pd

_WS_PATTERN = re.compile(r"\s+")

def normalize_str(s: Any, collapse_ws: bool = True) -> Any:
    """Safely normalize a scalar text for robust parsing/matching.

    - Unicode 정규화(NFKC)
    - BOM/제로폭 제거
    - NBSP/좁은 NBSP → 일반 공백
    - 스마트 따옴표 → ASCII 따옴표
    - (옵션) 연속 공백 축약 + strip

    Args:
        s: 임의 스칼라(문자열/숫자/None 등)
        collapse_ws: True면 연속 공백을 하나로 축약하고 좌우 공백 제거

    Returns:
        Any: 문자열이면 정규화된 문자열, 아니면 원본
    """
    if not isinstance(s, str):
        return s
    t = unicodedata.normalize("NFKC", s)
    # 보이지 않는/문제 유발 문자 정리
    t = (t.replace("\ufeff", "")    # BOM
           .replace("\u200b", "")   # ZWSP
           .replace("\u200c", "")   # ZWNJ
           .replace("\u200d", "")   # ZWJ
           .replace("\u2060", ""))  # WORD JOINER
    # 특수 공백 → 일반 공백
    t = (t.replace("\u00a0", " ")   # NBSP
           .replace("\u202f", " ")) # NARROW NBSP
    # 스마트 따옴표 → ASCII
    t = (t.replace("“", '"').replace("”", '"')
           .replace("‘", "'").replace("’", "'"))
    if collapse_ws:
        t = _WS_PATTERN.sub(" ", t).strip()
    return t


def read_csv_normalize_header(
    filepath_or_buffer: str | PathLike[str] | io.TextIOBase | io.BytesIO,
    /,
    *args: Any,
    **kwargs: Any,
) -> pd.DataFrame:
    """Drop-in 대체용 read_csv: 헤더 라인만 normalize_str 적용 후 파싱.

    pandas.read_csv와 동일하게 사용하되, header로 지정된 라인만 선처리한다.
    - header: int | list[int] | None | 'infer' 지원
    - encoding 기본값은 'utf-8-sig'로 가정(문자열/바이트 입력 모두 처리)
    - engine 미지정 시 python으로 기본 설정(깨진 CSV에 관대)

    Args:
        filepath_or_buffer: 파일 경로/버퍼(텍스트/바이트)
        *args: pd.read_csv의 원래 가변인자(그대로 전달)
        **kwargs: pd.read_csv의 키워드 인자(그대로 전달)

    Returns:
        pd.DataFrame: 파싱된 데이터프레임
    """
    # 0) header 인식(기본 'infer' → 0으로 간주해 헤더 라인 정규화)
    header = kwargs.get("header", "infer")
    if header == "infer":
        header_idx_list: list[int] = [0]
    elif isinstance(header, int):
        header_idx_list = [header]
    elif isinstance(header, list):
        header_idx_list = header
    else:
        header_idx_list = []  # None 등: 헤더 없음 → 정규화하지 않음

    # 1) 원문 읽기 (경로/버퍼 모두 지원)
    encoding = kwargs.get("encoding", "utf-8-sig")
    if isinstance(filepath_or_buffer, (str, os.PathLike)):
        with open(filepath_or_buffer, "rb") as f:
            raw_bytes = f.read()
        text = raw_bytes.decode(encoding, errors="replace")
    elif isinstance(filepath_or_buffer, io.BytesIO):
        pos = filepath_or_buffer.tell()
        filepath_or_buffer.seek(0)
        raw_bytes = filepath_or_buffer.read()
        filepath_or_buffer.seek(pos)
        text = raw_bytes.decode(encoding, errors="replace")
    else:
        # TextIOBase 등 텍스트 버퍼
        pos = filepath_or_buffer.tell() if hasattr(filepath_or_buffer, "tell") else None
        text = filepath_or_buffer.read()
        if hasattr(filepath_or_buffer, "seek") and pos is not None:
            filepath_or_buffer.seek(pos)

    # 2) 헤더 라인만 정규화 (collapse_ws=False: 내부 공백 유지, NBSP만 space로)
    lines = text.splitlines()
    for idx in header_idx_list:
        if 0 <= idx < len(lines):
            lines[idx] = normalize_str(lines[idx], collapse_ws=False)
    cleaned_text = "\n".join(lines)

    # 3) pandas로 파싱 (StringIO 사용 → encoding 키는 무의미하므로 제거)
    kwargs.pop("encoding", None)
    # engine 기본값: python(유연). 사용자가 명시했으면 존중
    kwargs.setdefault("engine", "python")

    return pd.read_csv(io.StringIO(cleaned_text), *args, **kwargs)
