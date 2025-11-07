from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List
import pandas as pd


# ==== Types & Constants ====

ColumnMap = Dict[str, str]

# 명시 라인 매핑: 규칙보다 우선
_EXPLICIT_LINE_MAP: Dict[str, int] = {"GI": 9, "GL": 12, "GM": 13}


# ==== Small, Focused Utilities ====

def _to_str_key(s: pd.Series) -> pd.Series:
    """키 Series를 문자열로 정규화.

    숫자형이면 소수점 제거 후 정수 문자열, 그 외는 문자열로 변환.
    NaN은 유지.

    Args:
        s: 키로 사용할 Series.

    Returns:
        문자열로 정규화된 Series.
    """
    if pd.api.types.is_numeric_dtype(s):
        # 숫자형 → 정수 → 문자열
        return s.apply(lambda x: None if pd.isna(x) else str(int(float(x))))
    return s.astype("string")


def _alpha_to_line(ch: str) -> float | int:
    """알파벳 한 글자를 추정 라인 번호로 변환.

    기준: I=9, 이후 알파벳 순서대로 +1.

    Args:
        ch: 알파벳 한 글자.

    Returns:
        추정 라인 번호(정수) 또는 NaN(float).
    """
    if isinstance(ch, str) and len(ch) == 1 and ch.isalpha():
        return ord(ch.upper()) - ord("I") + 9
    return float("nan")


def _infer_line_from_lot(heat_lot_no: pd.Series) -> pd.Series:
    """'heat_lot_no' 앞 두 글자로 라인 번호를 추정(Int64).

    규칙:
      1) 명시 매핑 우선: GI→9, GL→12, GM→13
      2) 그 외 G? 패턴은 두 번째 알파벳으로 I=9 기준 순번 추정
      3) 규칙 불가 시 NaN

    Args:
        heat_lot_no: LOT 번호 Series.

    Returns:
        라인 번호(Int64) Series.
    """
    s = heat_lot_no.astype("string").str.upper()

    # 1) 명시 매핑
    first2 = s.str[:2]
    explicit = first2.map(_EXPLICIT_LINE_MAP)

    # 2) G? 추정: 명시 미적용 + 'G'로 시작
    need_est = explicit.isna() & s.str.startswith("G")
    second_char = s.str[1]
    est = second_char.where(need_est).map(_alpha_to_line)

    # 3) 결합 및 Int64 캐스팅
    return explicit.fillna(est).astype("Int64")


def _prdt_to_long(prdt: pd.DataFrame, need_cols: Iterable[str]) -> pd.DataFrame:
    """PRDT 와이드 → 롱 변환(필요한 컬럼만).

    Args:
        prdt: PRDT 원본 DataFrame.
        need_cols: 변환 대상 컬럼 목록.

    Returns:
        열: ['입고LOT','prdt_col','품질검사값'] 의 long 포맷 DataFrame.
    """
    cols: List[str] = [c for c in pd.unique(pd.Series(list(need_cols))) if c in prdt.columns]
    if not cols:
        return pd.DataFrame(columns=["입고LOT", "prdt_col", "품질검사값"])
    # 와이드→롱 변환
    return prdt.melt(
        id_vars=["입고LOT"],
        value_vars=cols,
        var_name="prdt_col",
        value_name="품질검사값",
    )


# ==== Public API ====

def merge_op_qual_prdt(
    op_qual_path: str | Path,
    prdt_path: str | Path,
    col_map: ColumnMap,
) -> pd.DataFrame:
    """OP_QUAL(xlsx)과 PRDT(csv)를 LOT 기준으로 병합하고 '라인'을 추가.

    파이프라인:
      (1) 파일 로드
      (2) 키 정규화
      (3) '비고'→PRDT 컬럼명 매핑
      (4) PRDT 와이드→롱 변환
      (5) (heat_lot_no, prdt_col) 기준 병합
      (6) 최종 컬럼 정리 + '라인' 계산

    Args:
        op_qual_path: OP_QUAL 엑셀 절대경로.
        prdt_path: PRDT CSV 절대경로(utf-8-sig 가정).
        col_map: OP_QUAL의 '비고' → PRDT 컬럼명 매핑.

    Returns:
        열 순서:
        ['heat_lot_no','라인','비고','품질검사값','최적화전품질값','최적화후품질값']
    """
    # (1) 로드
    op = pd.read_excel(op_qual_path)
    pr = pd.read_csv(prdt_path, encoding="utf-8-sig")

    # (2) 키 정규화
    op = op.copy()
    pr = pr.copy()
    op["heat_lot_no"] = _to_str_key(op["heat_lot_no"])
    pr["입고LOT"] = _to_str_key(pr["입고LOT"])

    # (3) '비고' → PRDT 컬럼명
    op["prdt_col"] = op["비고"].map(col_map)

    # (4) PRDT 롱 변환(필요 컬럼만)
    pr_long = _prdt_to_long(pr, need_cols=op["prdt_col"].dropna())

    # (5) (heat_lot_no, prdt_col) 병합
    merged = op.merge(
        pr_long,
        left_on=["heat_lot_no", "prdt_col"],
        right_on=["입고LOT", "prdt_col"],
        how="left",
    )

    # (6) 최종 컬럼 + '라인'
    out = merged[["heat_lot_no", "비고", "품질검사값", "최적화전품질값", "최적화후품질값"]].copy()
    out.insert(1, "라인", _infer_line_from_lot(out["heat_lot_no"]))
    out = out.sort_values(["heat_lot_no", "비고"], kind="stable").reset_index(drop=True)
    return out
