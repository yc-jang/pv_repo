import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 데이터 읽기 (파일 경로는 실제 환경에 맞게 수정)
df_pred = pd.read_csv("out.csv")
df_actual = pd.read_csv("prdt.csv")

# 2. 데이터 병합 (Key: LOT번호 == 입고LOT)
# suffixes를 붙여서 혹시 모를 컬럼명 충돌 방지
merged_df = pd.merge(
    df_pred, 
    df_actual, 
    left_on="LOT번호", 
    right_on="입고LOT", 
    how="inner",
    suffixes=('_pred', '_real') 
)

# 3. 비교할 타겟 컬럼 쌍 찾기
# (out.csv의 'PRDT_' 컬럼 -> prdt.csv의 원래 컬럼 매칭)
target_pairs = []
for col in df_pred.columns:
    if col.startswith("PRDT_"):
        real_col_name = col.replace("PRDT_", "") # 접두어 제거
        
        # 실제 데이터에 해당 컬럼이 있는지 확인
        if real_col_name in df_actual.columns:
            target_pairs.append((real_col_name, col)) # (실제이름, 예측이름)

# 4. 시각화 (Subplots 생성)
# 타겟 개수에 맞춰서 자동으로 행/열 계산 (여기서는 2열로 배치)
n_plots = len(target_pairs)
rows = (n_plots + 1) // 2
cols = 2

fig = make_subplots(
    rows=rows, cols=cols, 
    subplot_titles=[f"{real} vs {pred}" for real, pred in target_pairs],
    horizontal_spacing=0.1, vertical_spacing=0.15
)

for idx, (real_col, pred_col) in enumerate(target_pairs):
    row = (idx // cols) + 1
    col = (idx % cols) + 1
    
    # (1) 산점도 추가 (Actual vs Predict)
    fig.add_trace(
        go.Scatter(
            x=merged_df[real_col], 
            y=merged_df[pred_col],
            mode='markers',
            name=real_col,
            marker=dict(size=6, opacity=0.6)
        ),
        row=row, col=col
    )
    
    # (2) 기준선 추가 (y=x) -> 예측이 정확할수록 이 선 위에 모임
    min_val = min(merged_df[real_col].min(), merged_df[pred_col].min())
    max_val = max(merged_df[real_col].max(), merged_df[pred_col].max())
    
    fig.add_shape(
        type="line",
        x0=min_val, y0=min_val, x1=max_val, y1=max_val,
        line=dict(color="Red", width=1, dash="dash"),
        row=row, col=col
    )
    
    # 축 라벨 설정
    fig.update_xaxes(title_text="Actual (실제)", row=row, col=col)
    fig.update_yaxes(title_text="Predicted (예측)", row=row, col=col)

# 5. 레이아웃 최종 다듬기
fig.update_layout(
    title_text="Model Predictions vs Actual Values Comparison",
    height=400 * rows,  # 그래프 개수에 따라 높이 자동 조절
    width=1000,
    showlegend=False    # 심플하게 보기 위해 범례 숨김
)

fig.show()



import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------
# 1. 데이터 로드 및 전처리
# ---------------------------------------------------------
# 파일 읽기
df_pred = pd.read_csv("out.csv")
df_actual = pd.read_csv("prdt.csv")

# 데이터 병합 (Key: LOT번호 == 입고LOT)
merged_df = pd.merge(
    df_pred, 
    df_actual, 
    left_on="LOT번호", 
    right_on="입고LOT", 
    how="inner",
    suffixes=('_pred', '_real') 
)

# [중요] Line Plot의 경향성을 보기 위해 LOT번호(시간/순서) 순으로 정렬
merged_df = merged_df.sort_values("LOT번호").reset_index(drop=True)

# ---------------------------------------------------------
# 2. 비교 대상 컬럼 쌍 찾기
# ---------------------------------------------------------
target_pairs = []
for col in df_pred.columns:
    if col.startswith("PRDT_"):
        real_col_name = col.replace("PRDT_", "") # 접두어 제거
        
        # 실제 데이터셋에 해당 컬럼이 존재하는지 확인
        if real_col_name in df_actual.columns:
            target_pairs.append((real_col_name, col)) # (실제이름, 예측이름)

# ---------------------------------------------------------
# 3. Figure 및 Subplot 설정
# ---------------------------------------------------------
# 타겟 하나당 2개의 그래프(Line + Scatter)가 필요하므로 행 개수는 * 2
n_targets = len(target_pairs)
total_rows = n_targets * 2 

# Subplot 제목 생성 (순서대로 Line -> Scatter)
subplot_titles = []
for real, pred in target_pairs:
    subplot_titles.append(f"[{real}] Trend Analysis (Line)")       # 홀수 행
    subplot_titles.append(f"[{real}] Correlation Check (Scatter)") # 짝수 행

fig = make_subplots(
    rows=total_rows, 
    cols=1, 
    subplot_titles=subplot_titles,
    vertical_spacing=0.08  # 그래프 간 간격 적절히 조절
)

# ---------------------------------------------------------
# 4. 그래프 그리기 (Loop)
# ---------------------------------------------------------
for idx, (real_col, pred_col) in enumerate(target_pairs):
    # 행 번호 계산 (Plotly는 1부터 시작)
    # idx=0 -> line_row=1, scatter_row=2
    # idx=1 -> line_row=3, scatter_row=4
    line_row = (idx * 2) + 1
    scatter_row = (idx * 2) + 2
    
    # --- [A] Line Plot (Trend) ---
    # 1. 실제값 Line
    fig.add_trace(
        go.Scatter(
            x=merged_df.index, 
            y=merged_df[real_col],
            mode='lines+markers',
            name=f'{real_col} (Actual)',
            line=dict(color='royalblue', width=2),
            marker=dict(size=4),
            legendgroup=f"group_{real_col}", # 같은 타겟끼리 범례 그룹화
        ),
        row=line_row, col=1
    )
    
    # 2. 예측값 Line
    fig.add_trace(
        go.Scatter(
            x=merged_df.index, 
            y=merged_df[pred_col],
            mode='lines+markers',
            name=f'{real_col} (Pred)',
            line=dict(color='firebrick', width=2, dash='dot'),
            marker=dict(size=4, symbol='x'),
            legendgroup=f"group_{real_col}",
        ),
        row=line_row, col=1
    )
    
    # --- [B] Scatter Plot (Correlation) ---
    # 3. 산점도 (X: 실제, Y: 예측)
    fig.add_trace(
        go.Scatter(
            x=merged_df[real_col],
            y=merged_df[pred_col],
            mode='markers',
            name=f'{real_col} (Corr)',
            marker=dict(color='purple', size=6, opacity=0.6),
            showlegend=False # 산점도는 범례 생략 (깔끔함을 위해)
        ),
        row=scatter_row, col=1
    )
    
    # 4. 기준선 (Red Dashed Line, y=x)
    min_val = min(merged_df[real_col].min(), merged_df[pred_col].min())
    max_val = max(merged_df[real_col].max(), merged_df[pred_col].max())
    
    fig.add_shape(
        type="line",
        x0=min_val, y0=min_val, x1=max_val, y1=max_val,
        line=dict(color="Red", width=1, dash="dash"),
        row=scatter_row, col=1
    )

    # --- [C] 축 라벨 설정 ---
    # Line Plot 축
    fig.update_yaxes(title_text="Value", row=line_row, col=1)
    
    # Scatter Plot 축
    fig.update_xaxes(title_text="Actual Value", row=scatter_row, col=1)
    fig.update_yaxes(title_text="Predicted Value", row=scatter_row, col=1)

# ---------------------------------------------------------
# 5. 레이아웃 최종 설정
# ---------------------------------------------------------
fig.update_layout(
    title_text="<b>Model Analysis Report: Trend & Correlation</b>",
    title_x=0.5, # 제목 중앙 정렬
    height=400 * total_rows, # 그래프 개수에 따라 전체 높이 자동 조절 (한 쌍당 800px)
    width=1000,              # 가로 폭 고정
    hovermode="x unified",   # Line Plot에서 마우스 오버 시 비교 편리
    showlegend=True
)

fig.show()
