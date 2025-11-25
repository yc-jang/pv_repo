# --- top_df 정렬 (score 기준, 작을수록 좋음) ---
top_sorted = top_df.sort_values(score_col).copy()
best_idx = top_sorted.index[0]
knee_idx = set(knee_df.index)

# 마스크 정의
is_best = top_sorted.index == best_idx               # top-1
is_knee = top_sorted.index.isin(knee_idx)            # knee 포함 여부

# 4가지 조합으로 나누기
top_best_knee     = top_sorted[ is_best &  is_knee]
top_best_only     = top_sorted[ is_best & ~is_knee]
top_rest_knee     = top_sorted[~is_best &  is_knee]
top_rest_only     = top_sorted[~is_best & ~is_knee]

# 공통 marker 설정 helper
def add_top_trace(df_sub, name, symbol, color, line_color, size):
    if df_sub.empty:
        return
    fig.add_trace(
        go.Scatter3d(
            x=df_sub[x],
            y=df_sub[y],
            z=df_sub[z],
            mode="markers",
            name=name,
            marker=dict(
                symbol=symbol,
                size=size,
                color=color,
                opacity=0.95,
                line=dict(color=line_color, width=3),
            ),
        )
    )

# 1) top-1 (knee 아님) : 파란 다이아
add_top_trace(
    df_sub=top_best_only,
    name="top-1",
    symbol="diamond",
    color="blue",
    line_color="rgba(0,0,0,0)",
    size=15,
)

# 2) top-1 (knee) : 파란 다이아 + 빨간 테두리
add_top_trace(
    df_sub=top_best_knee,
    name="top-1 (knee)",
    symbol="diamond",
    color="blue",
    line_color="red",
    size=15,
)

# 3) other top (not knee) : 초록 원
add_top_trace(
    df_sub=top_rest_only,
    name="top-n",
    symbol="circle",
    color="green",
    line_color="rgba(0,0,0,0)",
    size=10,
)

# 4) other top (knee) : 초록 원 + 빨간 테두리
add_top_trace(
    df_sub=top_rest_knee,
    name="top-n (knee)",
    symbol="circle",
    color="green",
    line_color="red",
    size=10,
)
