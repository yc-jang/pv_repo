import numpy as np
import pandas as pd

# ---------- 공통 유틸 ----------
def _norm_str(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    return np.nan if s == "" else s

def _collapse_consecutive_dups(parts):
    out, prev = [], object()
    for p in parts:
        if p != prev:
            out.append(p)
        prev = p
    return out

def read_header_from_clipboard() -> pd.DataFrame:
    """엑셀에서 헤더 블록(A1:E96 등) 복사 후 호출"""
    return pd.read_clipboard(sep="\t", header=None)

# ---------- (기존) 가로 경계 기반 복원: top-down + intra-segment horizontal ffill ----------
def fill_header_block_with_hier_boundaries(header_block: pd.DataFrame) -> pd.DataFrame:
    """
    윗행이 만든 '가로 경계(열 인덱스 경계)'를 누적하며, 세그먼트 내부에서만 좌->우 ffill.
    """
    H = header_block.copy().applymap(_norm_str)
    n_rows, n_cols = H.shape
    boundaries = [0, n_cols]                         # 열 기준 경계(0, n_cols 포함)
    F = pd.DataFrame(np.full((n_rows, n_cols), np.nan, dtype=object))

    for r in range(n_rows):
        row_vals = H.iloc[r, :].tolist()
        new_row = [np.nan] * n_cols
        for i in range(len(boundaries) - 1):
            s, e = boundaries[i], boundaries[i+1]
            seg = row_vals[s:e]
            last = np.nan
            for j in range(len(seg)):
                v = seg[j]
                if isinstance(v, str) and v != "":
                    last = v
                if isinstance(last, str) and last != "":
                    new_row[s + j] = last
        F.iloc[r, :] = new_row

        # 이 행에서 값이 바뀌는 지점을 다음 행의 경계로 추가
        new_boundaries = set(boundaries)
        prev = F.iat[r, 0]
        for c in range(1, n_cols):
            cur = F.iat[r, c]
            if (pd.isna(prev) and not pd.isna(cur)) or (not pd.isna(prev) and pd.isna(cur)) or (prev != cur):
                new_boundaries.add(c)
            prev = cur
        boundaries = sorted(new_boundaries)

    F.columns = header_block.columns
    F.index = header_block.index
    return F

# ---------- (신규) 세로 경계 기반 복원: left-right + intra-segment vertical ffill ----------
def fill_header_block_with_vertical_boundaries(header_block: pd.DataFrame) -> pd.DataFrame:
    """
    왼쪽 열부터 처리하며 '세로 경계(행 인덱스 경계)'를 누적.
    각 경계 세그먼트 내부에서만 위->아래 ffill.
    """
    H = header_block.copy().applymap(_norm_str)
    n_rows, n_cols = H.shape
    boundaries = [0, n_rows]                         # 행 기준 경계(0, n_rows 포함)
    F = pd.DataFrame(np.full((n_rows, n_cols), np.nan, dtype=object))

    for c in range(n_cols):
        col_vals = H.iloc[:, c].tolist()
        new_col = [np.nan] * n_rows
        for i in range(len(boundaries) - 1):
            s, e = boundaries[i], boundaries[i+1]
            seg = col_vals[s:e]
            last = np.nan
            for j in range(len(seg)):
                v = seg[j]
                if isinstance(v, str) and v != "":
                    last = v
                if isinstance(last, str) and last != "":
                    new_col[s + j] = last
        F.iloc[:, c] = new_col

        # 이 열에서 값이 바뀌는 지점을 다음 열의 '세로 경계'로 추가
        new_boundaries = set(boundaries)
        prev = F.iat[0, c]
        for r in range(1, n_rows):
            cur = F.iat[r, c]
            if (pd.isna(prev) and not pd.isna(cur)) or (not pd.isna(prev) and pd.isna(cur)) or (prev != cur):
                new_boundaries.add(r)
            prev = cur
        boundaries = sorted(new_boundaries)

    F.columns = header_block.columns
    F.index = header_block.index
    return F

# ---------- 평탄화 래퍼: direction에 따라 가로/세로 복원 후 join ----------
def build_flat_labels_from_hier_header(
    header_block: pd.DataFrame,
    direction: str = "column",        # "column" 또는 "row"
    sep: str = "_",
    drop_empty: bool = True,
    collapse_dup_levels: bool = True,
    dedup_names: bool = True,
    as_frame: bool = True,            # True면 DataFrame으로 반환(요청하신 96x1에 적합)
    name: str = "joined"
):
    """
    direction="column": 열 단위 해석(기존 flat_cols) → 길이 = n_cols
    direction="row"   : 행 단위 해석(요청: 96x1)     → 길이 = n_rows
    """
    if direction not in {"column", "row"}:
        raise ValueError("direction must be 'column' or 'row'")

    # 1) 경계 보존 복원
    if direction == "column":
        F = fill_header_block_with_hier_boundaries(header_block)
    else:
        F = fill_header_block_with_vertical_boundaries(header_block)

    # 2) 파트 모으기 및 join
    items = []
    if direction == "column":
        # 각 열: 위→아래
        for c in range(F.shape[1]):
            parts = [F.iat[r, c] for r in range(F.shape[0])]
            parts = [str(p).strip() for p in parts if (not drop_empty) or (pd.notna(p) and str(p).strip() != "")]
            if collapse_dup_levels:
                parts = _collapse_consecutive_dups(parts)
            label = sep.join([p for p in parts if p]) or f"col_{c}"
            items.append(label)
    else:
        # 각 행: 좌→우
        for r in range(F.shape[0]):
            parts = [F.iat[r, c] for c in range(F.shape[1])]
            parts = [str(p).strip() for p in parts if (not drop_empty) or (pd.notna(p) and str(p).strip() != "")]
            if collapse_dup_levels:
                parts = _collapse_consecutive_dups(parts)
            label = sep.join([p for p in parts if p]) or f"row_{r}"
            items.append(label)

    # 3) 중복 처리
    if dedup_names:
        seen, out = {}, []
        for nm in items:
            if nm not in seen:
                seen[nm] = 0
                out.append(nm)
            else:
                seen[nm] += 1
                out.append(f"{nm}.{seen[nm]}")
        items = out

    # 4) 반환 형태
    if direction == "column":
        return pd.Index(items, name=name) if not as_frame else pd.DataFrame({name: items})
    else:
        # 요청: 96x1 형태 → DataFrame 기본값(as_frame=True)
        return pd.DataFrame({name: items}) if as_frame else pd.Index(items, name=name)
