import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_e2e_from_csv(
    csv_path: str,
    index_col: str | None = None,
    title: str = "실측 vs 예측 (최적화 전/후)",
    save_html: str | None = None
):
    """
    CSV를 읽어 다음 컬럼을 사용해 인터랙티브 Plotly 그래프를 생성합니다.
      - PRDT_Target: 검사 실측값
      - Before_optim: 최적화 전 예측
      - After_Optim : 최적화 후 예측

    특징
    - 상단: 실측/예측 라인 비교
    - 하단: 잔차(residual) 비교 (After-PRDT, Before-PRDT)
    - RMSE/MAE 개선 효과를 레이아웃 상단에 요약 표기
    - 600개 포인트 기준 줌, 스파이크, 레인지슬라이더 활성화

    Parameters
    ----------
    csv_path : str
        CSV 파일 경로
    index_col : str | None
        x축으로 사용할 인덱스 컬럼명 (없으면 1..N 시퀀스 사용)
    title : str
        메인 타이틀
    save_html : str | None
        저장할 html 경로 (예: "result.html"). None이면 저장하지 않음.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    # 1) CSV 로드
    df = pd.read_csv(csv_path)

    # 2) 흔한 케이스 처리: 'Unnamed: 0' 등 불필요 인덱스 열 제거
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]

    # 3) 컬럼 정규화(공백 제거, 대소문자/언더스코어 통일)
    mapper = {c: c.strip() for c in df.columns}
    df = df.rename(columns=mapper)

    required = ["PRDT_Target", "Before_optim", "After_Optim"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV에 필수 컬럼이 없습니다: {missing} / 존재 컬럼: {list(df.columns)}")

    # 4) 숫자 변환 및 결측/비정상 제거
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required).reset_index(drop=True)

    # 5) x축 설정
    if index_col and index_col in df.columns:
        x = df[index_col]
        x_title = index_col
    else:
        x = np.arange(1, len(df) + 1)
        x_title = "Index"

    # 6) 지표 계산 (RMSE/MAE)
    y_true   = df["PRDT_Target"].to_numpy()
    y_before = df["Before_optim"].to_numpy()
    y_after  = df["After_Optim"].to_numpy()

    def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
    def mae(a, b):  return float(np.mean(np.abs(a - b)))

    rmse_b, rmse_a = rmse(y_true, y_before), rmse(y_true, y_after)
    mae_b,  mae_a  = mae(y_true, y_before),  mae(y_true, y_after)

    # 7) 서브플롯 구성
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.06,
        subplot_titles=("실측/예측 비교", "잔차 비교 (예측 - 실측)")
    )

    # ── (상단) 실측/예측 라인 ───────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=x, y=y_true, mode="lines+markers", name="PRDT_Target(실측)",
            hovertemplate=f"{x_title}=%{{x}}<br>실측=%{{y:.4f}}<extra></extra>"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=y_before, mode="lines", name="Before_optim(기존예측)",
            hovertemplate=f"{x_title}=%{{x}}<br>기존예측=%{{y:.4f}}<extra></extra>"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=y_after, mode="lines", name="After_Optim(최적화예측)",
            hovertemplate=f"{x_title}=%{{x}}<br>최적화예측=%{{y:.4f}}<extra></extra>"
        ),
        row=1, col=1
    )

    # ── (하단) 잔차(예측-실측) 비교 ──────────────────────────
    resid_before = y_before - y_true
    resid_after  = y_after  - y_true

    fig.add_trace(
        go.Scatter(
            x=x, y=resid_before, mode="lines", name="잔차: Before-PRDT",
            hovertemplate=f"{x_title}=%{{x}}<br>잔차(Before)=%{{y:.4f}}<extra></extra>"
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=resid_after, mode="lines", name="잔차: After-PRDT",
            hovertemplate=f"{x_title}=%{{x}}<br>잔차(After)=%{{y:.4f}}<extra></extra>"
        ),
        row=2, col=1
    )

    # 0 기준선
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="gray", row=2, col=1)

    # 8) 레이아웃/인터랙션
    subtitle = (
        f"RMSE: Before {rmse_b:.4f} → After {rmse_a:.4f} "
        f"(Δ {rmse_a - rmse_b:+.4f}) | "
        f"MAE: Before {mae_b:.4f} → After {mae_a:.4f} "
        f"(Δ {mae_a - mae_b:+.4f})"
    )

    fig.update_layout(
        title={"text": f"{title}<br><sup>{subtitle}</sup>", "x": 0.01},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
        hovermode="x unified",
        xaxis=dict(
            title=x_title,
            showspikes=True, spikemode="across", spikesnap="cursor", spikedash="dot"
        ),
        xaxis2=dict(
            title=x_title,
            showspikes=True, spikemode="across", spikesnap="cursor", spikedash="dot",
            rangeslider=dict(visible=True)  # 아래 축에 레인지슬라이더
        ),
        yaxis=dict(title="값"),
        yaxis2=dict(title="잔차 (예측 - 실측)")
    )

    # 9) HTML 저장 옵션
    if save_html:
        import plotly.io as pio
        pio.write_html(fig, file=save_html, include_plotlyjs="cdn", auto_open=False)

    return fig
