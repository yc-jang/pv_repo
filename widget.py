import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output
from typing import Any, Dict, List


try:
    import shap
except ImportError:  # pragma: no cover - optional dependency
    shap = None
    
class UserControlInputWidget:
    """사용자 입력을 받아 예측을 수행하는 인터랙티브 위젯."""

    def __init__(
        self,
        model: Any,
        user_control_columns: Dict[str, Dict[str, float]],
        reference: pd.DataFrame,
        highlight_columns: List[str] | None = None,
        user_dropdown_columns: List[str] | None = None,
        *,
        expansion_alpha: float = 2.0,
        clamp_min: float | None = None,
        clamp_max: float | None = None,
        dependent_columns: List[str] | None = None,
    ) -> None:
        """위젯을 초기화한다.

        Parameters
        ----------
        model : Any
            ``predict`` 메서드를 가진 예측 모델
        user_control_columns : dict
            ``int`` 또는 ``float`` 타입별로 컬럼과 스텝을 지정하는 딕셔너리
        reference : pandas.DataFrame
            기본 값과 범위 계산에 사용하는 참조 데이터
        highlight_columns : list[str], optional
            강조 표시할 컬럼 목록
        user_dropdown_columns : list[str], optional
            0/50/100 중 하나를 선택하는 드롭다운으로 사용할 컬럼 목록
        expansion_alpha : float, optional
            입력 허용 범위를 ``alpha * std`` 만큼 확장할 배수 (1.0~3.0)
        clamp_min : float, optional
            확장 시 적용할 최소값 제한
        clamp_max : float, optional
            확장 시 적용할 최대값 제한
        dependent_columns : list[str], optional
            다른 입력에 의해 계산되는 읽기 전용 컬럼
        """

        self.model = model
        self.user_control_columns = user_control_columns
        self.reference = reference
        self.highlight_columns = highlight_columns or []
        self.expansion_alpha = max(1.0, min(expansion_alpha, 3.0))
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.dependent_columns = dependent_columns or []
        self.user_dropdown_columns = user_dropdown_columns or []

        # 검색을 위한 기본 컬럼 이름
        self.lot_column = "lot번호"
        self.date_column = "날짜"
        
        # Keep columns attribute for compatibility with prediction stage
        int_cols = list(user_control_columns.get("int", {}).keys())
        float_cols = list(user_control_columns.get("float", {}).keys())
        dropdown_cols = list(self.user_dropdown_columns)
        self.columns = int_cols + float_cols + dropdown_cols
        self.total_df = pd.DataFrame(columns=self.columns)

        # User input widgets generated from provided columns
        self.widgets_dict = {}

        # Buttons
        self.submit_button = widgets.Button(description="Submit", button_style="success")
        self.predict_button = widgets.Button(description="Predict", button_style="info")
        self.reset_button = widgets.Button(description="Reset", button_style="warning")
        self.submit_button.on_click(self._on_submit)
        self.predict_button.on_click(self._on_predict)
        self.reset_button.on_click(self._on_reset)

        self.output = widgets.Output()
        self.df_output = widgets.Output()
        self.delete_output = widgets.Output()
        self.search_output = widgets.Output()


        # SHAP 시각화를 위한 위젯과 상태값
        self.index_choice_dropdown = widgets.Dropdown(description="SHAP Index", options=[])
        self.shap_plot_button = widgets.Button(description="SHAP Plot")
        self.shap_plot_button.on_click(self._on_shap_plot)
        self.shap_output = widgets.Output()
        self.model_input: pd.DataFrame | None = None
        self.shap_values: Any | None = None
        # 행 삭제를 위한 드롭다운과 버튼
        self.delete_dropdown = widgets.Dropdown(description="삭제 Index", options=[])
        self.delete_button = widgets.Button(description="Delete", button_style="danger")
        self.delete_button.on_click(self._on_delete)

        # 검색 위젯들
        self.start_date_picker = widgets.DatePicker(description="시작일")
        self.end_date_picker = widgets.DatePicker(description="종료일")
        self.lot_dropdown = widgets.Dropdown(description="Lot")

        self.start_date_picker.observe(self._on_date_change, names="value")
        self.end_date_picker.observe(self._on_date_change, names="value")
        self.lot_dropdown.observe(self._on_lot_change, names="value")


        self._build_ui()
        self._attach_observers()

    def _build_ui(self) -> None:
        """위젯 화면을 구성한다."""

        # 검색 박스
        self.lot_dropdown.options = self._get_lot_options()
        search_box = widgets.HBox(
            [self.start_date_picker, self.end_date_picker, self.lot_dropdown]
        )

        widget_list = []  # 각 입력 위젯을 차례로 담을 리스트
        for dtype, cols in self.user_control_columns.items():
            for col, step in cols.items():
                series = self.reference[col] if col in self.reference else pd.Series(dtype=float)
                mean_val = series.mean() if not series.empty else 0
                min_val = series.min() if not series.empty else 0
                max_val = series.max() if not series.empty else 0
                std_val = series.std() if not series.empty else 0

                expanded_min = min_val - self.expansion_alpha * std_val
                expanded_max = max_val + self.expansion_alpha * std_val
                if self.clamp_min is not None:
                    expanded_min = max(expanded_min, self.clamp_min)
                if self.clamp_max is not None:
                    expanded_max = min(expanded_max, self.clamp_max)

                label_style = 'color:blue; font-weight:bold;' if col in self.highlight_columns else ''
                label = widgets.HTML(
                    value=(
                        f"<span style='{label_style}'>{col}</span> "
                        f"(min: {min_val:.2f} ~ max: {max_val:.2f}, "
                        f"mean: {mean_val:.2f})"
                    ),
                    layout=widgets.Layout(width='500px')
                )

                if dtype == 'int':
                    input_widget = widgets.BoundedIntText(
                        value=int(round(mean_val)),
                        step=step,
                        min=int(round(expanded_min)),
                        max=int(round(expanded_max)),
                    )
                else:
                    input_widget = widgets.BoundedFloatText(
                        value=round(float(mean_val), 3),
                        step=step,
                        min=float(expanded_min),
                        max=float(expanded_max),
                    )

                if col in self.dependent_columns:
                    input_widget.disabled = True

                self.widgets_dict[col] = input_widget
                widget_list.append(widgets.HBox([label, input_widget]))

        # 0/50/100 선택용 드롭다운 위젯 추가
        for col in self.user_dropdown_columns:
            label_style = 'color:blue; font-weight:bold;' if col in self.highlight_columns else ''
            label = widgets.HTML(
                value=f"<span style='{label_style}'>{col}</span>",
                layout=widgets.Layout(width='500px')
            )
            dropdown = widgets.Dropdown(options=[0, 50, 100], value=0)
            self.widgets_dict[col] = dropdown
            widget_list.append(widgets.HBox([label, dropdown]))

        buttons = widgets.HBox([self.submit_button, self.predict_button, self.reset_button])
        delete_box = widgets.HBox([self.delete_dropdown, self.delete_button])
        shap_box = widgets.HBox([self.index_choice_dropdown, self.shap_plot_button])
        form = widgets.VBox(
            [search_box, self.search_output]
            + widget_list
            + [buttons, delete_box, shap_box, self.delete_output, self.output, self.shap_output, self.df_output]
        )
        display(form)

    def _attach_observers(self) -> None:
        """입력 값이 변할 때 종속 컬럼을 갱신하도록 이벤트를 연결한다."""
        for col, widget in self.widgets_dict.items():
            if col not in self.dependent_columns and col not in self.user_dropdown_columns:
                widget.observe(self._on_input_change, names="value")

        # Initial update to compute dependent columns
        self._update_dependent_columns()

    def _get_lot_options(self) -> List[str]:
        """선택된 날짜 범위에 해당하는 lot 목록을 반환한다."""
        df = self.reference
        start = self.start_date_picker.value
        end = self.end_date_picker.value
        if start is not None:
            df = df[df[self.date_column] >= pd.to_datetime(start)]
        if end is not None:
            df = df[df[self.date_column] <= pd.to_datetime(end)]
        if self.lot_column in df.columns:
            return sorted(df[self.lot_column].dropna().unique().tolist())
        return []

    def _on_date_change(self, change: dict) -> None:
        """날짜 변경 시 lot 목록을 업데이트한다."""
        self.lot_dropdown.options = self._get_lot_options()
        self._on_lot_change({})


    def _on_lot_change(self, change: dict) -> None:
        """선택된 lot의 데이터를 불러와 입력 위젯에 채운다."""
        lot = self.lot_dropdown.value
        df = self.reference
        start = self.start_date_picker.value
        end = self.end_date_picker.value
        if start is not None:
            df = df[df[self.date_column] >= pd.to_datetime(start)]
        if end is not None:
            df = df[df[self.date_column] <= pd.to_datetime(end)]
        if lot is not None:
            df = df[df[self.lot_column] == lot]
        with self.search_output:
            clear_output()
            if df.empty:
                print("검색 결과가 없습니다.")
                return
            row = df.iloc[0]
            for col, widget in self.widgets_dict.items():
                if col in row.index:
                    widget.value = row[col]
            print(f"{lot} 데이터가 로드되었습니다.")
        self._update_dependent_columns()

    def _calculate_dependent_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """현재 입력 값으로부터 종속 컬럼을 계산한다."""

        # 예: 상단 + 중단 + 하단의 합을 총합 컬럼에 반영
        if {'상단', '중단', '하단', '총합'}.issubset(df.columns):
            df['총합'] = df[['상단', '중단', '하단']].sum(axis=1)

        return df

    def _validate_input(self, df: pd.DataFrame) -> List[str]:
        """사용자 입력이 규칙을 지키는지 확인한다."""

        errors: List[str] = []
        if '총합' in df.columns and (df['총합'] != 16).any():
            errors.append("'총합'은 반드시 16이어야 합니다.")
        return errors

    def _style_dataframe(self) -> None:
        """검증 결과를 반영해 데이터프레임을 꾸민다."""

        # 유효성 검사를 위한 하이라이트 함수
        def highlight_invalid(val):
            return 'background-color: red' if val != 16 else ''

        styled_df = self.total_df.style

        # Highlight 총합 validity if present
        if '총합' in self.total_df.columns:
            styled_df = styled_df.applymap(highlight_invalid, subset=['총합'])

        # Highlight user specified columns based on reference statistics
        for col in self.highlight_columns:
            if col in self.total_df.columns and col in self.reference.columns:
                mean = self.reference[col].mean()
                std = self.reference[col].std()

                def highlight_range(val, m=mean, s=std):
                    return 'background-color: yellow' if val < m - s or val > m + s else ''

                styled_df = styled_df.applymap(highlight_range, subset=[col])

        with self.df_output:
            clear_output()
            display(styled_df)

    def _on_submit(self, b: widgets.Button) -> None:
        """사용자가 입력한 값을 데이터프레임에 추가한다."""

        # 모든 입력 값을 하나의 딕셔너리로 모은다
        input_data = {col: widget.value for col, widget in self.widgets_dict.items()}
        input_df = pd.DataFrame([input_data], columns=self.columns)

        # Calculate dependent columns
        input_df = self._calculate_dependent_columns(input_df)

        # Validate input
        errors = self._validate_input(input_df)

        # Add to total DataFrame
        self.total_df = pd.concat([self.total_df, input_df], ignore_index=True)

        # 새 행이 추가되었으므로 삭제 드롭다운 옵션을 갱신한다
        self._update_delete_box()
        self._style_dataframe()
        # 모델 입력 및 SHAP 값은 새 예측 전까지 무효화한다
        self.model_input = None
        self.shap_values = None
        self.index_choice_dropdown.options = []

        with self.output:
            clear_output()
            if errors:
                for err in errors:
                    print(f"Validation Error: {err}")

    def _update_delete_box(self) -> None:
        """삭제 드롭다운 옵션을 최신 상태로 유지한다."""
        self.delete_dropdown.options = list(range(len(self.total_df)))

    def _update_dependent_columns(self) -> None:
        """현재 입력 값으로 종속 컬럼을 다시 계산하고 위젯에 반영한다."""
        input_values = {c: w.value for c, w in self.widgets_dict.items()}
        df = pd.DataFrame([input_values], columns=self.columns)
        df = self._calculate_dependent_columns(df)

        for col in self.dependent_columns:
            if col in df.columns and col in self.widgets_dict:
                val = df[col].iloc[0]
                widget = self.widgets_dict[col]
                widget.value = val
                # 종속 컬럼 값을 해당 위젯에 반영한다

    def _on_input_change(self, change: dict) -> None:
        """입력 값 변경 시 호출되어 종속 컬럼을 갱신한다."""

        self._update_dependent_columns()

    def _on_reset(self, b: widgets.Button) -> None:
        """모든 입력 결과를 초기 상태로 되돌린다."""
        self.total_df = pd.DataFrame(columns=self.columns)
        self._update_delete_box()
        self._style_dataframe()
        self.model_input = None
        self.shap_values = None
        self.index_choice_dropdown.options = []

    def _delete_row(self, idx: int) -> None:
        """지정한 행을 삭제하고 버튼 상태를 갱신한다."""

        # 범위를 벗어나면 아무 것도 하지 않음
        if idx >= len(self.total_df):
            return

        self.total_df = self.total_df.drop(index=idx).reset_index(drop=True)
        self._update_delete_box()
        self._style_dataframe()
        self.model_input = None
        self.shap_values = None
        self.index_choice_dropdown.options = []

    def _on_delete(self, b: widgets.Button) -> None:
        """드롭다운에서 선택된 행을 삭제한다."""
        if self.delete_dropdown.value is None:
            return
        self._delete_row(int(self.delete_dropdown.value))

    def _on_predict(self, b: widgets.Button) -> None:
        """모은 데이터를 사용하여 모델 예측을 실행한다."""


        with self.output:
            clear_output()
            if self.total_df.empty:
                print("예측할 데이터가 없습니다.")
                return

            # Validate data before prediction
            errors = self._validate_input(self.total_df)
            if errors:
                print("예측 전 해결해야 할 유효성 오류가 있습니다:")
                for err in errors:
                    print(f"- {err}")
                return

            feature_cols = [c for c in self.total_df.columns if c != 'Prediction']

            self.model_input = self.total_df[feature_cols].copy()
            predictions = self.model.predict(self.model_input)
            result_df = self.total_df.copy()
            result_df['Prediction'] = predictions

            print("예측 결과:")
            display(result_df)

            # SHAP 값을 계산하고 인덱스 선택 옵션을 제공
            if shap is not None:
                self.shap_values = shap.Explainer(self.model)(self.model_input)
                self.index_choice_dropdown.options = list(range(len(self.model_input)))

    def _on_shap_plot(self, b: widgets.Button) -> None:
        """선택된 행의 SHAP 워터폴 그래프를 표시한다."""
        with self.shap_output:
            clear_output()
            if shap is None:
                print("shap 라이브러리가 설치되어 있지 않습니다.")
                return
            if self.shap_values is None or self.index_choice_dropdown.value is None:
                print("예측을 먼저 수행해 주세요.")
                return

            idx = int(self.index_choice_dropdown.value)
            shap.plots.waterfall(self.shap_values[idx])

            # 입력 값과 참조 평균을 함께 표시해 비교 가이드를 제공
            row = self.model_input.iloc[[idx]]
            means = self.reference[row.columns].mean().to_frame().T
            display(pd.concat([row, means], keys=["선택값", "평균"], axis=0))

# Example usage (Replace with actual model and reference data)
# xgb_model = trained_xgb_model
# reference_df = pd.read_csv('training_data.csv')
# control_columns = {
#     'int': {'하단': 1, '중단': 1, '상단': 1, '총합': 1},
# }
# widget = UserControlInputWidget(
#     model=xgb_model,
#     user_control_columns=control_columns,
#     reference=reference_df,
#     highlight_columns=list(control_columns.get('int', {}).keys()),
# )
