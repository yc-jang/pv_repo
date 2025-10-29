from __future__ import annotations
import csv
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import pandas as pd
import numpy as np

def _open_sample(p: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Try UTF-8-SIG → CP949, return small text sample."""
    for enc in ("utf-8-sig", "cp949"):
        try:
            with p.open("r", encoding=enc, errors="strict") as f:
                return f.read(4096), enc, None
        except Exception as e: 
            last_err = f"{type(e).__name__}({enc}): {e}"
    return None, None, last_err

def _detect_dialect(sample: str) -> Tuple[Optional[csv.Dialect], Optional[str]]:
    """Sniff CSV dialect from sample text."""
    try:
        return csv.Sniffer().sniff(sample), None
    except Exception as e:
        return None, f"SniffError: {e}"

def _lint_csv_file(p: Path, max_rows: Optional[int] = None) -> List[str]:
    """Return problem messages for a single CSV file (empty list if ok)."""
    probs: List[str] = []
    sample, enc, err = _open_sample(p)
    if sample is None or enc is None:
        probs.append(err or "EncodingError")
        return probs

    # Empty / whitespace-only file check
    if not sample.strip():
        try:
            if not p.read_text(encoding=enc).strip():
                probs.append("EmptyFile: no content"); return probs
        except Exception as e:
            probs.append(f"ReadError: {e}"); return probs

    dialect, sniff_err = _detect_dialect(sample)
    if dialect is None: probs.append(sniff_err or "DialectDetectError")

    # Fallback defaults if sniffing failed
    delimiter = (dialect.delimiter if dialect else ",")
    quotechar = (dialect.quotechar if dialect and dialect.quotechar else '"')
    doublequote = (dialect.doublequote if dialect else True)
    skipinitialspace = (dialect.skipinitialspace if dialect else False)
    quoting = (dialect.quoting if dialect else csv.QUOTE_MINIMAL)

    row_count = 0
    header: Optional[List[str]] = None
    expected_cols: Optional[int] = None
    bad_rows: List[int] = []

    try:
        with p.open("r", encoding=enc, newline="") as f:
            reader = csv.reader(
                f, delimiter=delimiter, quotechar=quotechar,
                doublequote=doublequote, skipinitialspace=skipinitialspace,
                quoting=quoting
            )
            for i, row in enumerate(reader, start=1):
                # Skip blank rows
                if not row or all((not str(c).strip() for c in row)):
                    continue
                row_count += 1
                if header is None:
                    header = row; expected_cols = len(row)
                else:
                    if expected_cols is not None and len(row) != expected_cols:
                        bad_rows.append(i)
                if max_rows is not None and row_count >= max_rows:
                    break
    except csv.Error as e:
        probs.append(f"CSVParseError: {e}"); return probs
    except Exception as e:
        probs.append(f"ReadError: {e}"); return probs

    if row_count == 0:
        probs.append("NoDataRows: no non-empty data rows found")

    if bad_rows:
        head = bad_rows[:5]
        tail = "..." if len(bad_rows) > 5 else ""
        probs.append(f"InconsistentColumns: expected {expected_cols}, mismatch at rows {head}{tail}")

    if header:
        names = [str(c).strip() for c in header]
        empties = [i for i, n in enumerate(names) if n == ""]
        if empties: probs.append(f"EmptyHeaderNames: positions {empties}")
        counts: Dict[str, int] = {}
        for n in names: counts[n] = counts.get(n, 0) + 1
        dups = [n for n, c in counts.items() if c > 1 and n != ""]
        if dups: probs.append(f"DuplicateHeaderNames: {dups}")

    return probs

def collect_csv_lint_report(root_dir: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Lint all CSVs under root and return a DataFrame report.

    Args:
        root_dir: Root directory path.
        max_rows: Optional per-file max rows to scan for performance.

    Returns:
        pd.DataFrame: columns = ['path', 'problem_count', 'problems', 'problems_str'].
    """
    root = Path(root_dir)
    records: List[Dict[str, object]] = []

    # 하위 *.csv 순회
    for p in root.rglob("*.csv"):
        probs = _lint_csv_file(p, max_rows=max_rows)
        if probs:  # 문제 있는 파일만 리포트에 포함
            records.append({
                "path": str(p),
                "problem_count": len(probs),
                "problems": probs,
                "problems_str": "; ".join(probs)
            })

    df = pd.DataFrame.from_records(records, columns=["path", "problem_count", "problems", "problems_str"])
    # 문제 많은 순, 경로순 정렬
    if not df.empty:
        df = df.sort_values(["problem_count", "path"], ascending=[False, True], ignore_index=True)
    return df
