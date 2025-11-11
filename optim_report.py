from __future__ import annotations
from typing import Dict, List, Tuple, Sequence, Optional
import numpy as np
import pandas as pd


# =========================
# 숫자/스펙 파싱 유틸
# =========================

def _extract_numeric_series(s: pd.Series) -> pd.Series:
    """문자 섞인 수치에서도 숫자만 안전 추출(예: '<0.01','ND','0.45%').
    - 핵심 원리: 정규식으로 부호/소수 포함 숫자 패턴만 추출 → float 변환
    """
    return (
        s.astype(str)
         .str.extract(r"([-+]?\d*\.?\d+)", expand=False)
         .astype(float)
    )


def _parse_spec_item(item: Sequence) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """spec의 [low, desired, upper] → (low, desired, upper) 숫자 튜플."""
    def _to_num(x):
        if x is None:
            return None
        v = pd.to_numeric(x, errors="coerce")
        return None if pd.isna(v) else float(v)

    low     = _to_num(item[0]) if len(item) > 0 else None
    desired = _to_num(item[1]) if len(item) > 1 else None
    upper   = _to_num(item[2]) if len(item) > 2 else None
    return low, desired, upper


def _spec_width(low: Optional[float], upper: Optional[float]) -> Optional[float]:
    """스펙폭(U-L). 하나라도 없거나 U<L이면 None."""
    if (low is None) or (upper is None):
        return None
    if upper < low:
        return None
    return upper - low if upper > low else 0.0


# =========================
# 지표 계산 함수 (비고 그룹 단위)
# =========================

def _central_proximity_stats(
    before: pd.Series,
    after: pd.Series,
    desired: Optional[float],
    spec_w: Optional[float],
    eps: float = 1e-12,
) -> dict:
    """중심 접근도: desired 대비 평균 편차의 절대값 감소율.

    정의(정규화된 절대편차):
      dev_b = |mean(before) - desired| / scale
      dev_a = |mean(after)  - desired| / scale
      scale = (U-L) if 가능, 아니면 결합 표준편차로 대체
    개선비율(gain):
      mean_gain = (dev_b - dev_a) / (dev_b + eps)
      → before를 기준으로 after가 얼마나 **비율로** 개선되었는지 (+면 개선, -면 악화)
    """
    b = _extract_numeric_series(before).to_numpy(dtype=float)
    a = _extract_numeric_series(after).to_numpy(dtype=float)

    if desired is None or np.isnan(desired):
        return dict(
            mean_편차_전=np.nan, mean_편차_후=np.nan,
            평균목표편차_전_정규화=np.nan, 평균목표편차_후_정규화=np.nan,
            평균목표접근개선율=np.nan,
        )

    diff_b = float(np.nanmean(b) - desired)
    diff_a = float(np.nanmean(a) - desired)

    if spec_w is not None and spec_w > 0:
        dev_b = abs(diff_b) / spec_w
        dev_a = abs(diff_a) / spec_w
    else:
        pooled = np.concatenate([b[~np.isnan(b)], a[~np.isnan(a)]])
        sd = float(np.nanstd(pooled, ddof=1))
        dev_b = abs(diff_b) if (sd == 0 or np.isnan(sd)) else abs(diff_b) / sd
        dev_a = abs(diff_a) if (sd == 0 or np.isnan(sd)) else abs(diff_a) / sd

    mean_gain = (dev_b - dev_a) / (dev_b + eps)

    return dict(
        mean_편차_전=diff_b,
        mean_편차_후=diff_a,
        평균목표편차_전_정규화=dev_b,
        평균목표편차_후_정규화=dev_a,
        평균목표접근개선율=mean_gain,
    )


def _stability_stats(
    before: pd.Series,
    after: pd.Series,
    spec_w: Optional[float],
    eps: float = 1e-12,
) -> dict:
    """분산 안정성: 표준편차 감소율.

    정의:
      std_gain = 1 - (std_after / (std_before + eps))
      → before 대비 after의 **상대 감소율** (+면 산포 감소=안정화)
    보고값:
      - 표준편차(전/후), 정규화 표준편차(전/후: (U-L)로 나눈 값)
    """
    b = _extract_numeric_series(before).to_numpy(dtype=float)
    a = _extract_numeric_series(after).to_numpy(dtype=float)

    std_b = float(np.nanstd(b, ddof=1))
    std_a = float(np.nanstd(a, ddof=1))

    if spec_w is not None and spec_w > 0:
        std_b_n = std_b / spec_w
        std_a_n = std_a / spec_w
    else:
        std_b_n, std_a_n = std_b, std_a

    std_gain = 1.0 - (std_a / (std_b + eps))

    return dict(
        표준편차_전=std_b,
        표준편차_후=std_a,
        표준편차_전_정규화=std_b_n,
        표준편차_후_정규화=std_a_n,
        산포개선율=std_gain,
    )


def _rmse_stats(
    before: pd.Series,
    after: pd.Series,
    desired: Optional[float],
    spec_w: Optional[float],
    eps: float = 1e-12,
) -> dict:
    """오차 척도: desired 기준 RMSE 감소율.

    정의:
      rmse_b = sqrt(mean((before - desired)^2))
      rmse_a = sqrt(mean((after  - desired)^2))
      rmse_gain = (rmse_b - rmse_a) / (rmse_b + eps)
      → before 대비 after의 **상대 감소율** (+면 오차 감소=개선)
    보고값:
      - RMSE(전/후), 정규화 RMSE(전/후: (U-L)로 나눈 값)
    """
    b = _extract_numeric_series(before).to_numpy(dtype=float)
    a = _extract_numeric_series(after).to_numpy(dtype=float)

    if desired is None or np.isnan(desired):
        return dict(
            RMSE_전=np.nan, RMSE_후=np.nan,
            RMSE_전_정규화=np.nan, RMSE_후_정규화=np.nan,
            RMSE개선율=np.nan,
        )

    rmse_b = float(np.sqrt(np.nanmean((b - desired) ** 2)))
    rmse_a = float(np.sqrt(np.nanmean((a - desired) ** 2)))

    if spec_w is not None and spec_w > 0:
        rmse_b_n = rmse_b / spec_w
        rmse_a_n = rmse_a / spec_w
    else:
        rmse_b_n, rmse_a_n = rmse_b, rmse_a

    rmse_gain = (rmse_b - rmse_a) / (rmse_b + eps)

    return dict(
        RMSE_전=rmse_b,
        RMSE_후=rmse_a,
        RMSE_전_정규화=rmse_b_n,
        RMSE_후_정규화=rmse_a_n,
        RMSE개선율=rmse_gain,
    )


def _in_spec_stats(
    before: pd.Series,
    after: pd.Series,
    low: Optional[float],
    upper: Optional[float],
) -> dict:
    """스펙 내 적합도: (L ≤ x ≤ U) 비율 및 개선(퍼센트포인트).

    정의:
      p_before = in-spec 비율(전), p_after = in-spec 비율(후)
      in_spec_pp = (p_after - p_before)   # 퍼센트포인트 차이(0~1 범위를 나중에 100x)
      → before 대비 after의 **절대 차이** (+면 합격률 상승)
    """
    if (low is None) or (upper is None) or (upper < low):
        return dict(
            스펙내율_전=np.nan, 스펙내율_후=np.nan, 스펙내율개선_pp=np.nan,
        )

    b = _extract_numeric_series(before)
    a = _extract_numeric_series(after)

    p_b = float(((b >= low) & (b <= upper)).mean())
    p_a = float(((a >= low) & (a <= upper)).mean())

    return dict(
        스펙내율_전=p_b,
        스펙내율_후=p_a,
        스펙내율개선_pp=p_a - p_b,
    )


# =========================
# 메인: 비고별 계산 → “값만” 담긴 보고용 DataFrame
# =========================

def compute_quality_stats_report(
    out: pd.DataFrame,
    spec: Dict[str, Sequence],
    *,
    col_before: str = "최적화전품질값",
    col_after: str  = "최적화후품질값",
    eps: float = 1e-12,
) -> pd.DataFrame:
    """비고별 4대 지표를 **값만** 담은 한국어 컬럼 DataFrame으로 반환.

    포함 항목(모두 값만; 판단 로직 없음):
      - 스펙: 스펙하한, 목표값, 스펙상한, 스펙폭
      - 중심 접근도: mean_편차(전/후), 평균목표편차_전/후_정규화, 평균목표접근개선율
      - 분산 안정성: 표준편차(전/후), 표준편차_전/후_정규화, 산포개선율
      - 오차 척도: RMSE(전/후), RMSE_전/후_정규화, RMSE개선율
      - 스펙내 적합도: 스펙내율(전/후), 스펙내율개선(pp)
      - 보조: 샘플수(전/후)
    """
    rows: List[dict] = []

    for remark, g in out.groupby("비고", dropna=False):
        key = str(remark)
        low, desired, upper = (None, None, None)
        if key in spec:
            low, desired, upper = _parse_spec_item(spec[key])

        spec_w = _spec_width(low, upper)

        # 원시 값 추출(샘플수 보고용)
        b_raw = _extract_numeric_series(g[col_before])
        a_raw = _extract_numeric_series(g[col_after])

        cp  = _central_proximity_stats(b_raw, a_raw, desired, spec_w, eps=eps)
        st  = _stability_stats(b_raw, a_raw, spec_w, eps=eps)
        rm  = _rmse_stats(b_raw, a_raw, desired, spec_w, eps=eps)
        isp = _in_spec_stats(b_raw, a_raw, low, upper)

        row = {
            "비고": remark,
            "스펙하한": low, "목표값": desired, "스펙상한": upper, "스펙폭": spec_w,
            # 중심 접근도
            "mean_편차_전": cp["mean_편차_전"],
            "mean_편차_후": cp["mean_편차_후"],
            "평균목표편차_전_정규화": cp["평균목표편차_전_정규화"],
            "평균목표편차_후_정규화": cp["평균목표편차_후_정규화"],
            "평균목표접근개선율": cp["평균목표접근개선율"],  # (dev_b - dev_a) / dev_b
            # 분산 안정성
            "표준편차_전": st["표준편차_전"],
            "표준편차_후": st["표준편차_후"],
            "표준편차_전_정규화": st["표준편차_전_정규화"],
            "표준편차_후_정규화": st["표준편차_후_정규화"],
            "산포개선율": st["산포개선율"],              # 1 - std_after/std_before
            # 오차 척도
            "RMSE_전": rm["RMSE_전"],
            "RMSE_후": rm["RMSE_후"],
            "RMSE_전_정규화": rm["RMSE_전_정규화"],
            "RMSE_후_정규화": rm["RMSE_후_정규화"],
            "RMSE개선율": rm["RMSE개선율"],              # (rmse_b - rmse_a)/rmse_b
            # 스펙 내 적합도
            "스펙내율_전": isp["스펙내율_전"],
            "스펙내율_후": isp["스펙내율_후"],
            "스펙내율개선_pp": isp["스펙내율개선_pp"],   # p_after - p_before
            # 보조
            "샘플수_전": int(b_raw.notna().sum()),
            "샘플수_후": int(a_raw.notna().sum()),
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("비고").reset_index(drop=True)

    # 가독성: 스펙내율은 %로 보고할 일이 많아 보조 컬럼 추가(원본은 0~1 유지)
    df["스펙내율_전(%)"] = (df["스펙내율_전"] * 100).astype(float)
    df["스펙내율_후(%)"] = (df["스펙내율_후"] * 100).astype(float)
    df["스펙내율개선(pp)"] = (df["스펙내율개선_pp"] * 100).astype(float)

    # 최종 열 순서(값만, 판단 없음)
    ordered_cols = [
        "비고",
        "스펙하한", "목표값", "스펙상한", "스펙폭",
        "평균목표편차_전_정규화", "평균목표편차_후_정규화", "평균목표접근개선율",
        "표준편차_전_정규화", "표준편차_후_정규화", "산포개선율",
        "RMSE_전_정규화", "RMSE_후_정규화", "RMSE개선율",
        "스펙내율_전(%)", "스펙내율_후(%)", "스펙내율개선(pp)",
        "mean_편차_전", "mean_편차_후", "표준편차_전", "표준편차_후", "RMSE_전", "RMSE_후",
        "스펙내율_전", "스펙내율_후", "스펙내율개선_pp",
        "샘플수_전", "샘플수_후",
    ]
    return df[ordered_cols]

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# =========================
# Keep existing structure
# =========================

def _ensure_dir(p: Path) -> None:
    """Create directory if not exists."""
    p.mkdir(parents=True, exist_ok=True)


def _safe_filename(text: str) -> str:
    """Keep filename-safe characters only."""
    return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in str(text))


def _melt_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    """Wide→long for plotting three metrics into single 'value' column."""
    cols = ["품질검사값", "최적화전품질값", "최적화후품질값"]
    present = [c for c in cols if c in df.columns]
    return df.melt(
        id_vars=["heat_lot_no", "라인", "비고"],
        value_vars=present,
        var_name="metric",
        value_name="value",
    )


# =========================
# New helpers (shapes + y-range lock)
# =========================

def _parse_spec(spec_item: Sequence) -> Tuple[float | None, float | None, float | None]:
    """Parse [low, desired, upper] into numeric tuple."""
    def _num(x):
        return None if x is None else pd.to_numeric(x, errors="coerce")
    low = _num(spec_item[0]) if len(spec_item) > 0 else None
    desired = _num(spec_item[1]) if len(spec_item) > 1 else None
    upper = _num(spec_item[2]) if len(spec_item) > 2 else None
    return (None if pd.isna(low) else float(low),
            None if pd.isna(desired) else float(desired),
            None if pd.isna(upper) else float(upper))


def _add_spec_shapes(
    fig: go.Figure,
    x_vals: List[str],
    low: float | None,
    desired: float | None,
    upper: float | None,
    band_color: str = "rgba(0,200,0,0.15)",
    target_color: str = "green",
) -> None:
    """Add spec band(line-independent) and target line via layout.shapes (no autorange impact)."""
    if not x_vals:
        return
    x0, x1 = x_vals[0], x_vals[-1]

    # 허용 밴드(rect)
    if (low is not None) and (upper is not None) and (upper >= low):
        fig.add_shape(
            type="rect", xref="x", yref="y",
            x0=x0, x1=x1, y0=low, y1=upper,
            line=dict(width=0),
            fillcolor=band_color,
            layer="below",
        )

    # 목표선(line)
    if desired is not None:
        fig.add_shape(
            type="line", xref="x", yref="y",
            x0=x0, x1=x1, y0=desired, y1=desired,
            line=dict(color=target_color, width=2),
            layer="above",
        )


def _lock_y_range_to_data(
    fig: go.Figure,
    y_values: np.ndarray,
    pad_ratio: float = 0.08,
) -> None:
    """Fix y-axis to data range (optional) so far-away specs won't compress traces."""
    y = y_values[~np.isnan(y_values)]
    if y.size == 0:
        return
    y_min, y_max = float(np.min(y)), float(np.max(y))
    pad = (y_max - y_min) * pad_ratio if y_max > y_min else max(1e-6, y_max * pad_ratio)
    fig.update_yaxes(range=[y_min - pad, y_max + pad])


# =========================
# Drop-in: keep names & flow
# =========================

def _plot_single_group(
    melted: pd.DataFrame,
    title: str,
    spec: Dict[str, List] | None = None,
    lock_y_to_data: bool = True,
) -> "px.Figure":
    """Per-remark line chart with optional spec band/target via shapes."""
    # LOT 순서를 카테고리 순서로 고정
    order = pd.unique(melted["heat_lot_no"]).tolist()
    base = melted.sort_values("heat_lot_no", kind="stable")

    fig = px.line(
        base, x="heat_lot_no", y="value", color="metric",
        markers=True, category_orders={"heat_lot_no": order}, title=title,
    )
    fig.update_layout(
        xaxis_title="heat_lot_no",
        yaxis_title="Value",
        legend_title="Metric",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
    )

    # spec을 shapes로 오버레이(축 자동범위에 영향 없음)
    if spec is not None:
        remark_key = str(melted["비고"].iloc[0])
        if remark_key in spec:
            low, desired, upper = _parse_spec(spec[remark_key])
            _add_spec_shapes(fig, order, low, desired, upper)

    # 선택: 실제 데이터 범위 기준으로 y축 고정
    if lock_y_to_data:
        _lock_y_range_to_data(fig, base["value"].to_numpy(), pad_ratio=0.08)

    return fig


def export_quality_plots(
    out_df: pd.DataFrame,
    by_line: bool = True,
    output_dir: str | Path = ".",
    width: int = 1280,
    height: int = 720,
    scale: int = 2,
    spec: Dict[str, List] | None = None,   # ⬅️ 추가: remark → [low, desired, upper]
    lock_y_to_data: bool = True,           # ⬅️ 추가: 축 고정 옵션
) -> None:
    """Export per-remark Plotly charts as PNG, with optional spec band/target (shapes)."""
    df = out_df.copy()
    base_dir = Path(output_dir).resolve()
    _ensure_dir(base_dir)

    melted = _melt_for_plot(df)
    if melted.empty:
        return

    if by_line:
        for line, g_line in melted.groupby("라인", dropna=False):
            line_label = "NaN" if pd.isna(line) else str(int(line))
            line_dir = base_dir / f"라인_{_safe_filename(line_label)}"
            _ensure_dir(line_dir)

            for remark, g in g_line.groupby("비고", dropna=False):
                remark_label = "NaN" if pd.isna(remark) else str(remark)
                title = f"[라인 {line_label}] 비고: {remark_label}"
                fig = _plot_single_group(g, title, spec=spec, lock_y_to_data=lock_y_to_data)
                fname = f"line_{line_label}__remark_{_safe_filename(remark_label)}.png"
                fig.write_image(str(line_dir / fname), width=width, height=height, scale=scale)
    else:
        for remark, g in melted.groupby("비고", dropna=False):
            remark_label = "NaN" if pd.isna(remark) else str(remark)
            title = f"[전체 데이터] 비고: {remark_label}"
            fig = _plot_single_group(g, title, spec=spec, lock_y_to_data=lock_y_to_data)
            fname = f"total__remark_{_safe_filename(remark_label)}.png"
            fig.write_image(str(base_dir / fname), width=width, height=height, scale=scale)





from __future__ import annotations
from typing import Dict, List, Union
import pandas as pd


def find_oos_lots(
    out: pd.DataFrame,
    spec: Dict[str, List[Union[str, float, int]]],
) -> pd.DataFrame:
    """SPEC(비고별 상/하한) 기준으로 벗어난 LOT만 반환.

    규칙:
      - spec[key][0] = 상한(upper), spec[key][2] = 하한(lower) (없으면 무시)
      - 상한/하한 중 존재하는 경계만 검사
      - 비교는 '초과/미만' (동일값은 정상으로 간주)

    Args:
        out: merge_op_qual_prdt 결과. 열: 'heat_lot_no','비고','품질검사값' 포함.
        spec: 비고 → [상한, *, 하한] 형식의 딕셔너리 (문자/숫자 혼재 가능)

    Returns:
        out-of-spec 행만 담은 DataFrame (열: 'heat_lot_no','비고','품질검사값','하한','상한').
    """
    # 1) 비고→상/하한 매핑 사전 생성(결측/비정상 값은 NaN 처리)
    def _to_num(x):
        return pd.to_numeric(x, errors="coerce")

    upper_map = {k: _to_num(v[0]) if len(v) > 0 else pd.NA for k, v in spec.items()}
    lower_map = {k: _to_num(v[2]) if len(v) > 2 else pd.NA for k, v in spec.items()}

    df = out.copy()

    # 2) 경계값 및 측정값 수치화
    df["상한"] = df["비고"].map(upper_map)
    df["하한"] = df["비고"].map(lower_map)
    df["품질검사값_num"] = pd.to_numeric(df["품질검사값"], errors="coerce")

    # 3) out-of-spec 마스크 (존재하는 경계만 검사)
    mask_upper = df["상한"].notna() & (df["품질검사값_num"] > df["상한"])
    mask_lower = df["하한"].notna() & (df["품질검사값_num"] < df["하한"])
    mask = mask_upper | mask_lower

    # 4) 결과 정리(필요 컬럼만, 정렬)
    res = df.loc[mask, ["heat_lot_no", "비고", "품질검사값", "하한", "상한"]]
    return res.sort_values(["비고", "heat_lot_no"], kind="stable").reset_index(drop=True)


from __future__ import annotations
from typing import Dict
import pandas as pd


def find_oos_lots(out: pd.DataFrame, spec: Dict[str, list[str]]) -> pd.DataFrame:
    """SPEC 기준(비고별 상·하한)으로 out-of-spec LOT 목록을 반환한다.

    SPEC 형식:
      - key: out['비고'] 값
      - value: list[str]이며 value[0] = 상한, value[2] = 하한 (문자 가능)

    Args:
        out: merge_op_qual_prdt 결과 DataFrame. 열: 'heat_lot_no','비고','품질검사값' 포함.
        spec: 비고 → [상한, *, 하한] 형태의 사전.

    Returns:
        벗어난 LOT들만 담은 DataFrame (열: 'heat_lot_no','비고','품질검사값','하한','상한').
    """
    # SPEC → DataFrame (비고 인덱스, 상/하한 숫자화)
    spec_df = pd.DataFrame.from_dict(spec, orient="index").rename(columns={0: "상한", 2: "하한"})[["상한", "하한"]]
    spec_df["상한"] = pd.to_numeric(spec_df["상한"], errors="coerce")
    spec_df["하한"] = pd.to_numeric(spec_df["하한"], errors="coerce")

    # out ←→ SPEC 조인 및 숫자 변환
    merged = out.merge(spec_df, left_on="비고", right_index=True, how="left").copy()
    merged["품질검사값_num"] = pd.to_numeric(merged["품질검사값"], errors="coerce")

    # 벗어남 조건: 값 > 상한 or 값 < 하한
    mask = (merged["품질검사값_num"] > merged["상한"]) | (merged["품질검사값_num"] < merged["하한"])

    # 결과 정리
    cols = ["heat_lot_no", "비고", "품질검사값", "하한", "상한"]
    return merged.loc[mask, cols].sort_values(["비고", "heat_lot_no"], kind="stable").reset_index(drop=True)


from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List
import pandas as pd
import plotly.express as px


# =========================
# Small, Focused Utilities
# =========================

_EXPLICIT_LINE_MAP: Dict[str, int] = {"GI": 9, "GL": 12, "GM": 13}


def _ensure_dir(p: Path) -> None:
    """Create directory if not exists."""
    p.mkdir(parents=True, exist_ok=True)


def _safe_filename(text: str) -> str:
    """Keep filename-safe characters only."""
    return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in str(text))


def _to_str_key(s: pd.Series) -> pd.Series:
    """Normalize key Series to string.

    If numeric → cast to int (drop decimal) → string; else → string as-is.
    NaN preserved.
    """
    if pd.api.types.is_numeric_dtype(s):
        return s.apply(lambda x: None if pd.isna(x) else str(int(float(x))))
    return s.astype("string")


def _alpha_to_line(ch: str) -> float | int:
    """Map single alphabet to line number with I=9 baseline."""
    if isinstance(ch, str) and len(ch) == 1 and ch.isalpha():
        return ord(ch.upper()) - ord("I") + 9
    return float("nan")


def _infer_line_from_lot(heat_lot_no: pd.Series) -> pd.Series:
    """Infer line number(Int64) from first two chars of LOT.

    Rules:
      1) Explicit map first: GI→9, GL→12, GM→13
      2) For 'G?' pattern not mapped, use second letter with I=9 baseline
      3) Otherwise NaN
    """
    s = heat_lot_no.astype("string").str.upper()
    first2 = s.str[:2]
    explicit = first2.map(_EXPLICIT_LINE_MAP)
    need_est = explicit.isna() & s.str.startswith("G")
    second = s.str[1]
    est = second.where(need_est).map(_alpha_to_line)
    return explicit.fillna(est).astype("Int64")


def _prdt_to_long(prdt: pd.DataFrame, need_cols: Iterable[str]) -> pd.DataFrame:
    """Convert PRDT wide → long with needed columns only."""
    cols: List[str] = [c for c in pd.unique(pd.Series(list(need_cols))) if c in prdt.columns]
    if not cols:
        return pd.DataFrame(columns=["입고LOT", "prdt_col", "품질검사값"])
    # 와이드 → 롱 변환
    return prdt.melt(
        id_vars=["입고LOT"],
        value_vars=cols,
        var_name="prdt_col",
        value_name="품질검사값",
    )


def _melt_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    """Wide → long for plotting three metrics into single 'value' column."""
    value_cols = ["품질검사값", "최적화전품질값", "최적화후품질값"]
    present = [c for c in value_cols if c in df.columns]
    return df.melt(
        id_vars=["heat_lot_no", "라인", "비고"],
        value_vars=present,
        var_name="metric",
        value_name="value",
    )


def _plot_single_group(melted: pd.DataFrame, title: str) -> "px.Figure":
    """Build per-remark line chart."""
    # 핵심 로직: LOT 발생 순서를 카테고리 순서로 사용
    order = pd.unique(melted["heat_lot_no"])
    fig = px.line(
        melted.sort_values("heat_lot_no", kind="stable"),
        x="heat_lot_no", y="value", color="metric",
        markers=True,
        category_orders={"heat_lot_no": list(order)},
        title=title,
    )
    fig.update_layout(
        xaxis_title="heat_lot_no",
        yaxis_title="Value",
        legend_title="Metric",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


# =========================
# Public API
# =========================

def merge_op_qual_prdt(
    op_qual_path: str | Path,
    prdt_path: str | Path,
    col_map: Dict[str, str],
) -> pd.DataFrame:
    """Merge OP_QUAL(xlsx) and PRDT(csv) on LOT and add inferred '라인'.

    Pipeline:
      (1) Load files
      (2) Normalize keys to string
      (3) Map OP_QUAL['비고'] to PRDT columns via `col_map`
      (4) Convert PRDT wide→long for needed columns
      (5) Merge on (heat_lot_no, prdt_col)
      (6) Select final columns and compute '라인'

    Args:
        op_qual_path: Absolute path to OP_QUAL Excel.
        prdt_path: Absolute path to PRDT CSV (assume utf-8-sig).
        col_map: Mapping from OP_QUAL '비고' to PRDT column names.

    Returns:
        DataFrame with columns:
        ['heat_lot_no','라인','비고','품질검사값','최적화전품질값','최적화후품질값']
    """
    # (1) Load
    op = pd.read_excel(op_qual_path)
    pr = pd.read_csv(prdt_path, encoding="utf-8-sig")

    # (2) Normalize keys
    op = op.copy(); pr = pr.copy()
    op["heat_lot_no"] = _to_str_key(op["heat_lot_no"])
    pr["입고LOT"] = _to_str_key(pr["입고LOT"])

    # (3) Map remark → PRDT column
    op["prdt_col"] = op["비고"].map(col_map)

    # (4) PRDT long
    pr_long = _prdt_to_long(pr, need_cols=op["prdt_col"].dropna())

    # (5) Merge on (heat_lot_no, prdt_col)
    merged = op.merge(
        pr_long,
        left_on=["heat_lot_no", "prdt_col"],
        right_on=["입고LOT", "prdt_col"],
        how="left",
    )

    # (6) Final columns + '라인'
    out = merged[["heat_lot_no", "비고", "품질검사값", "최적화전품질값", "최적화후품질값"]].copy()
    out.insert(1, "라인", _infer_line_from_lot(out["heat_lot_no"]))
    out = out.sort_values(["heat_lot_no", "비고"], kind="stable").reset_index(drop=True)
    return out


def export_quality_plots(
    out_df: pd.DataFrame,
    by_line: bool = True,
    output_dir: str | Path = ".",
    width: int = 1280,
    height: int = 720,
    scale: int = 2,
) -> None:
    """Export per-remark Plotly charts as PNG images.

    If `by_line=True`, saves into subfolders per line (e.g., ./라인_9/).
    If `by_line=False`, saves into `output_dir` for entire dataset.

    Args:
        out_df: The DataFrame returned by `merge_op_qual_prdt`.
        by_line: Save per line subfolders if True; otherwise save as total.
        output_dir: Root directory to save results (default ".").
        width: Image width in px (default 1280).
        height: Image height in px (default 720).
        scale: Image scaling factor for write_image (default 2).

    Notes:
        - PNG export requires Plotly's static image engine (e.g., kaleido).
    """
    df = out_df.copy()
    base_dir = Path(output_dir).resolve()
    _ensure_dir(base_dir)

    melted = _melt_for_plot(df)
    if melted.empty:
        return

    if by_line:
        for line, g_line in melted.groupby("라인", dropna=False):
            line_label = "NaN" if pd.isna(line) else str(int(line))
            line_dir = base_dir / f"라인_{_safe_filename(line_label)}"
            _ensure_dir(line_dir)

            for remark, g in g_line.groupby("비고", dropna=False):
                remark_label = "NaN" if pd.isna(remark) else str(remark)
                title = f"[라인 {line_label}] 비고: {remark_label}"
                fig = _plot_single_group(g, title)
                fname = f"line_{line_label}__remark_{_safe_filename(remark_label)}.png"
                fig.write_image(str(line_dir / fname), width=width, height=height, scale=scale)
    else:
        for remark, g in melted.groupby("비고", dropna=False):
            remark_label = "NaN" if pd.isna(remark) else str(remark)
            title = f"[전체 데이터] 비고: {remark_label}"
            fig = _plot_single_group(g, title)
            fname = f"total__remark_{_safe_filename(remark_label)}.png"
            fig.write_image(str(base_dir / fname), width=width, height=height, scale=scale)


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


from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List
import pandas as pd
import plotly.express as px


# =========================
# Small, Focused Utilities
# =========================

_EXPLICIT_LINE_MAP: Dict[str, int] = {"GI": 9, "GL": 12, "GM": 13}


def _ensure_dir(p: Path) -> None:
    """Create directory if not exists."""
    p.mkdir(parents=True, exist_ok=True)


def _safe_filename(text: str) -> str:
    """Keep filename-safe characters only."""
    return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in str(text))


def _to_str_key(s: pd.Series) -> pd.Series:
    """Normalize key Series to string.

    If numeric → cast to int (drop decimal) → string; else → string as-is.
    NaN preserved.
    """
    if pd.api.types.is_numeric_dtype(s):
        return s.apply(lambda x: None if pd.isna(x) else str(int(float(x))))
    return s.astype("string")


def _alpha_to_line(ch: str) -> float | int:
    """Map single alphabet to line number with I=9 baseline."""
    if isinstance(ch, str) and len(ch) == 1 and ch.isalpha():
        return ord(ch.upper()) - ord("I") + 9
    return float("nan")


def _infer_line_from_lot(heat_lot_no: pd.Series) -> pd.Series:
    """Infer line number(Int64) from first two chars of LOT.

    Rules:
      1) Explicit map first: GI→9, GL→12, GM→13
      2) For 'G?' pattern not mapped, use second letter with I=9 baseline
      3) Otherwise NaN
    """
    s = heat_lot_no.astype("string").str.upper()
    first2 = s.str[:2]
    explicit = first2.map(_EXPLICIT_LINE_MAP)
    need_est = explicit.isna() & s.str.startswith("G")
    second = s.str[1]
    est = second.where(need_est).map(_alpha_to_line)
    return explicit.fillna(est).astype("Int64")


def _prdt_to_long(prdt: pd.DataFrame, need_cols: Iterable[str]) -> pd.DataFrame:
    """Convert PRDT wide → long with needed columns only."""
    cols: List[str] = [c for c in pd.unique(pd.Series(list(need_cols))) if c in prdt.columns]
    if not cols:
        return pd.DataFrame(columns=["입고LOT", "prdt_col", "품질검사값"])
    # 와이드 → 롱 변환
    return prdt.melt(
        id_vars=["입고LOT"],
        value_vars=cols,
        var_name="prdt_col",
        value_name="품질검사값",
    )


def _melt_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    """Wide → long for plotting three metrics into single 'value' column."""
    value_cols = ["품질검사값", "최적화전품질값", "최적화후품질값"]
    present = [c for c in value_cols if c in df.columns]
    return df.melt(
        id_vars=["heat_lot_no", "라인", "비고"],
        value_vars=present,
        var_name="metric",
        value_name="value",
    )


def _plot_single_group(melted: pd.DataFrame, title: str) -> "px.Figure":
    """Build per-remark line chart."""
    # 핵심 로직: LOT 발생 순서를 카테고리 순서로 사용
    order = pd.unique(melted["heat_lot_no"])
    fig = px.line(
        melted.sort_values("heat_lot_no", kind="stable"),
        x="heat_lot_no", y="value", color="metric",
        markers=True,
        category_orders={"heat_lot_no": list(order)},
        title=title,
    )
    fig.update_layout(
        xaxis_title="heat_lot_no",
        yaxis_title="Value",
        legend_title="Metric",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


# =========================
# Public API
# =========================

def merge_op_qual_prdt(
    op_qual_path: str | Path,
    prdt_path: str | Path,
    col_map: Dict[str, str],
) -> pd.DataFrame:
    """Merge OP_QUAL(xlsx) and PRDT(csv) on LOT and add inferred '라인'.

    Pipeline:
      (1) Load files
      (2) Normalize keys to string
      (3) Map OP_QUAL['비고'] to PRDT columns via `col_map`
      (4) Convert PRDT wide→long for needed columns
      (5) Merge on (heat_lot_no, prdt_col)
      (6) Select final columns and compute '라인'

    Args:
        op_qual_path: Absolute path to OP_QUAL Excel.
        prdt_path: Absolute path to PRDT CSV (assume utf-8-sig).
        col_map: Mapping from OP_QUAL '비고' to PRDT column names.

    Returns:
        DataFrame with columns:
        ['heat_lot_no','라인','비고','품질검사값','최적화전품질값','최적화후품질값']
    """
    # (1) Load
    op = pd.read_excel(op_qual_path)
    pr = pd.read_csv(prdt_path, encoding="utf-8-sig")

    # (2) Normalize keys
    op = op.copy(); pr = pr.copy()
    op["heat_lot_no"] = _to_str_key(op["heat_lot_no"])
    pr["입고LOT"] = _to_str_key(pr["입고LOT"])

    # (3) Map remark → PRDT column
    op["prdt_col"] = op["비고"].map(col_map)

    # (4) PRDT long
    pr_long = _prdt_to_long(pr, need_cols=op["prdt_col"].dropna())

    # (5) Merge on (heat_lot_no, prdt_col)
    merged = op.merge(
        pr_long,
        left_on=["heat_lot_no", "prdt_col"],
        right_on=["입고LOT", "prdt_col"],
        how="left",
    )

    # (6) Final columns + '라인'
    out = merged[["heat_lot_no", "비고", "품질검사값", "최적화전품질값", "최적화후품질값"]].copy()
    out.insert(1, "라인", _infer_line_from_lot(out["heat_lot_no"]))
    out = out.sort_values(["heat_lot_no", "비고"], kind="stable").reset_index(drop=True)
    return out


def export_quality_plots(
    out_df: pd.DataFrame,
    by_line: bool = True,
    output_dir: str | Path = ".",
    width: int = 1280,
    height: int = 720,
    scale: int = 2,
) -> None:
    """Export per-remark Plotly charts as PNG images.

    If `by_line=True`, saves into subfolders per line (e.g., ./라인_9/).
    If `by_line=False`, saves into `output_dir` for entire dataset.

    Args:
        out_df: The DataFrame returned by `merge_op_qual_prdt`.
        by_line: Save per line subfolders if True; otherwise save as total.
        output_dir: Root directory to save results (default ".").
        width: Image width in px (default 1280).
        height: Image height in px (default 720).
        scale: Image scaling factor for write_image (default 2).

    Notes:
        - PNG export requires Plotly's static image engine (e.g., kaleido).
    """
    df = out_df.copy()
    base_dir = Path(output_dir).resolve()
    _ensure_dir(base_dir)

    melted = _melt_for_plot(df)
    if melted.empty:
        return

    if by_line:
        for line, g_line in melted.groupby("라인", dropna=False):
            line_label = "NaN" if pd.isna(line) else str(int(line))
            line_dir = base_dir / f"라인_{_safe_filename(line_label)}"
            _ensure_dir(line_dir)

            for remark, g in g_line.groupby("비고", dropna=False):
                remark_label = "NaN" if pd.isna(remark) else str(remark)
                title = f"[라인 {line_label}] 비고: {remark_label}"
                fig = _plot_single_group(g, title)
                fname = f"line_{line_label}__remark_{_safe_filename(remark_label)}.png"
                fig.write_image(str(line_dir / fname), width=width, height=height, scale=scale)
    else:
        for remark, g in melted.groupby("비고", dropna=False):
            remark_label = "NaN" if pd.isna(remark) else str(remark)
            title = f"[전체 데이터] 비고: {remark_label}"
            fig = _plot_single_group(g, title)
            fname = f"total__remark_{_safe_filename(remark_label)}.png"
            fig.write_image(str(base_dir / fname), width=width, height=height, scale=scale)


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
