import numpy as np
import pandas as pd

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
    """엑셀에서 헤더 블록(A1:E96 등)을 복사한 뒤 호출"""
    df = pd.read_clipboard(sep="\t", header=None)
    return df

def fill_header_block_with_hier_boundaries(header_block: pd.DataFrame) -> pd.DataFrame:
    """
    계층 경계를 보존하면서 행단위 ffill을 수행.
    - 윗행이 만든 경계(라벨 전환점)를 아래행에서 '넘지 못하게' 하여 사용.
    - 클립보드 헤더만으로 복잡 병합을 안전하게 복원.
    """
    H = header_block.copy()
    H = H.applymap(_norm_str)
    n_rows, n_cols = H.shape

    # boundaries: 열 인덱스 경계들(0과 n_cols 포함)
    boundaries = [0, n_cols]
    F = pd.DataFrame(np.full((n_rows, n_cols), np.nan, dtype=object))

    for r in range(n_rows):
        # 현재 행 값
        row_vals = H.iloc[r, :].tolist()

        # 세그먼트별로 좌->우 ffill (경계를 넘지 않도록)
        new_row = [np.nan] * n_cols
        for i in range(len(boundaries) - 1):
            s, e = boundaries[i], boundaries[i+1]
            seg = row_vals[s:e]
            seg = [np.nan if (isinstance(x, float) and np.isnan(x)) else x for x in seg]
            # ffill within segment
            last = np.nan
            for j in range(len(seg)):
                if isinstance(seg[j], str) and seg[j] != "":
                    last = seg[j]
                elif not (isinstance(seg[j], float) and np.isnan(seg[j])):
                    last = seg[j]  # 안전용
                else:
                    # seg[j] is NaN => keep last
                    pass
                if isinstance(last, str) and last != "":
                    new_row[s + j] = last
                else:
                    new_row[s + j] = np.nan

        # 채운 행을 반영
        F.iloc[r, :] = new_row

        # 이번 행에서 라벨 전환이 일어난 지점을 새로운 경계로 추가
        new_boundaries = set(boundaries)
        prev = F.iat[r, 0]
        for c in range(1, n_cols):
            cur = F.iat[r, c]
            # 값이 달라지는 지점을 경계로 본다(한쪽이 NaN이어도 경계로 처리)
            if (pd.isna(prev) and not pd.isna(cur)) or (not pd.isna(prev) and pd.isna(cur)) or (prev != cur):
                new_boundaries.add(c)
            prev = cur

        boundaries = sorted(new_boundaries)

    F.columns = header_block.columns
    F.index = header_block.index
    return F

def build_flat_colnames_from_hier_header(
    header_block: pd.DataFrame,
    sep: str = "_",
    drop_empty=True,
    dedup_names=True,
    collapse_dup_levels=True
) -> pd.Index:
    """
    1) 헤더 블록을 계층 경계 보존 방식으로 가로 채움
    2) 각 열에 대해 위→아래 유효 라벨들을 모아 '_'로 join
    """
    F = fill_header_block_with_hier_boundaries(header_block)

    names = []
    n_rows, n_cols = F.shape
    for c in range(n_cols):
        parts = []
        for r in range(n_rows):
            v = F.iat[r, c]
            if drop_empty and (pd.isna(v) or (isinstance(v, str) and v.strip() == "")):
                continue
            parts.append(str(v).strip() if v is not None else "")
        if collapse_dup_levels:
            parts = _collapse_consecutive_dups(parts)
        parts = [p for p in parts if p != ""]  # 최종 공백 제거
        name = sep.join(parts) if parts else f"col_{c}"
        names.append(name)

    if dedup_names:
        seen = {}
        out = []
        for nm in names:
            if nm not in seen:
                seen[nm] = 0
                out.append(nm)
            else:
                seen[nm] += 1
                out.append(f"{nm}.{seen[nm]}")
        names = out

    return pd.Index(names)
