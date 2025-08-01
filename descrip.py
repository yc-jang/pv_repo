import numpy as np
import pandas as pd

# ---------- 유틸 ----------
def _is_numeric_like(x) -> bool:
    if x is None:
        return False
    if isinstance(x, (int, float, np.number)):
        return True
    try:
        float(str(x).replace(",", ""))
        return True
    except Exception:
        return False

def infer_header_depth(df: pd.DataFrame, probe_rows: int = 10, focus_cols=None) -> int:
    """
    상단 몇 행이 헤더인지 휴리스틱 추정.
    - 문자열 비율 높고 숫자 비율 낮은 행을 헤더로 간주
    """
    n = min(probe_rows, len(df))
    if focus_cols is None:
        focus_cols = list(range(min(5, df.shape[1])))
    depth = 0
    for r in range(n):
        row = df.iloc[r, focus_cols]
        vals = list(row)
        if len(vals) == 0:
            break
        num_ratio = np.mean([_is_numeric_like(v) for v in vals])
        str_ratio = np.mean([(isinstance(v, str) and str(v).strip() != "") for v in vals])
        if str_ratio >= 0.3 and num_ratio < 0.5:
            depth += 1
        else:
            break
    return max(depth, 1)  # 최소 1행은 헤더로 본다

def infer_stub_width(df: pd.DataFrame, max_check: int = 5, look_rows: int = 30) -> int:
    """
    좌측 스텁(행 헤더) 폭 추정: 왼쪽에서부터 숫자형 비율이 낮은 열을 카운트.
    """
    rows = min(look_rows, df.shape[0])
    width = 0
    for c in range(min(max_check, df.shape[1])):
        col = df.iloc[:rows, c]
        num_ratio = np.mean([_is_numeric_like(v) for v in col])
        if num_ratio <= 0.4:
            width += 1
        else:
            break
    return width

def flatten_multiindex_columns(
    df: pd.DataFrame,
    sep: str = "_",
    dedup: bool = True,
    strip: bool = True
) -> pd.DataFrame:
    """
    컬럼(MultiIndex/Index)을 평탄화하여 단일 Index로.
    각 레벨의 NaN/빈칸은 제거하고 sep로 join.
    """
    new_cols = []
    if isinstance(df.columns, pd.MultiIndex):
        for tup in df.columns:
            parts = []
            for x in tup:
                if pd.isna(x):
                    continue
                s = str(x)
                if strip:
                    s = s.strip()
                if s != "":
                    parts.append(s)
            name = sep.join(parts) if parts else "col"
            new_cols.append(name)
    else:
        for x in df.columns:
            s = str(x).strip() if strip else str(x)
            new_cols.append(s if s else "col")

    if dedup:
        seen = {}
        out = []
        for nm in new_cols:
            if nm not in seen:
                seen[nm] = 0
                out.append(nm)
            else:
                seen[nm] += 1
                out.append(f"{nm}.{seen[nm]}")
        new_cols = out

    out_df = df.copy()
    out_df.columns = pd.Index(new_cols)
    return out_df

# ---------- 메인 파이프라인 ----------
def process_clipboard_A1_E96_to_wide_and_long(
    override_header_depth: int = None,
    override_stub_width: int = None,
    sep: str = "_"
):
    """
    1) 엑셀에서 A1:E96 선택 후 Ctrl+C
    2) 이 함수를 실행하면 wide(df_wide)와 long(df_long) 반환
    """
    # 1) 클립보드에서 원본 그리드 읽기
    raw = pd.read_clipboard(sep="\t", header=None)
    raw = raw.replace("", np.nan)

    # 2) 헤더 깊이 / 스텁 폭 추정 (필요 시 오버라이드)
    header_depth = override_header_depth or infer_header_depth(raw, focus_cols=list(range(min(5, raw.shape[1]))))
    stub_width   = override_stub_width   or infer_stub_width(raw)

    # 3) 헤더 영역(상단 header_depth행)만 좌->우 ffill로 복원
    if header_depth > 0:
        raw.iloc[:header_depth, :] = raw.iloc[:header_depth, :].ffill(axis=1)

    # 4) 좌측 스텁 영역(컬럼 0~stub_width-1)만 위->아래 ffill (헤더 아래부터)
    if stub_width > 0:
        raw.iloc[header_depth:, :stub_width] = raw.iloc[header_depth:, :stub_width].ffill(axis=0)

    # 5) 멀티헤더 구성 → 본문 분리
    header_block = raw.iloc[:header_depth, :].copy()
    body = raw.iloc[header_depth:, :].reset_index(drop=True).copy()

    # MultiIndex columns 만들기
    arrays = []
    for r in range(header_depth):
        arrays.append([str(x).strip() if pd.notna(x) else "" for x in header_block.iloc[r]])
    if arrays:
        mi_cols = pd.MultiIndex.from_arrays(arrays, names=[f"level_{i+1}" for i in range(header_depth)])
    else:
        mi_cols = pd.RangeIndex(body.shape[1])
    body.columns = mi_cols

    # 6) 컬럼 평탄화(헤더 레벨들을 '_'로 join)
    df_wide = flatten_multiindex_columns(body, sep=sep)

    # 7) long 포맷으로 변환: 스텁을 id_vars로, 나머지 열을 변수화
    id_vars = df_wide.columns[:stub_width] if stub_width > 0 else []
    value_vars = df_wide.columns[stub_width:] if stub_width < df_wide.shape[1] else []
    df_long = df_wide.melt(id_vars=id_vars, value_vars=value_vars,
                           var_name="variable", value_name="value")

    meta = dict(header_depth=header_depth, stub_width=stub_width, n_rows=df_wide.shape[0], n_cols=df_wide.shape[1])
    return df_wide, df_long, meta
