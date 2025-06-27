import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import HTML, display

# 테스트용 데이터프레임 (컬럼 25개 정도로 테스트)
df = pd.DataFrame(np.random.rand(1, 25), columns=[f"col{i+1}" for i in range(25)])

def render_fixed_width_rows(df: pd.DataFrame, columns_per_block: int = 10, col_width_px: int = 100) -> str:
    html = """
    <style>
    .row-block {
        margin-bottom: 12px;
    }
    .grid-row {
        display: grid;
        grid-template-columns: """ + f"repeat({columns_per_block}, {col_width_px}px);" + """
        gap: 2px;
        font-size: 12px;
        word-break: break-word;
        overflow-wrap: break-word;
    }
    .cell {
        padding: 6px;
        text-align: center;
        border: 1px solid #ccc;
        background: #f5f5f5;
    }
    .cell.header {
        font-weight: bold;
        background: #dce9f9;
    }
    </style>
    """

    row = df.iloc[0]  # 단일 행 출력 가정

    total_columns = df.shape[1]
    num_blocks = (total_columns + columns_per_block - 1) // columns_per_block

    for block_idx in range(num_blocks):
        start = block_idx * columns_per_block
        end = min(start + columns_per_block, total_columns)
        col_subset = df.columns[start:end]
        val_subset = row.values[start:end]

        html += '<div class="row-block">'
        html += '<div class="grid-row">'
        for col in col_subset:
            html += f'<div class="cell header">{col}</div>'
        html += '</div>'

        html += '<div class="grid-row">'
        for val in val_subset:
            html += f'<div class="cell">{val}</div>'
        html += '</div>'
        html += '</div>'

    return html

# 위젯 출력
model_output = widgets.Output()
with model_output:
    display(HTML(render_fixed_width_rows(df, columns_per_block=10)))

display(model_output)
