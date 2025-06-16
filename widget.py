import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output

class UserControlInputWidget:
    """Interactive widget for collecting user input and performing prediction."""

    def __init__(self, model, user_control_columns, reference: pd.DataFrame, highlight_columns=None):
        """Create the widget.

        Parameters
        ----------
        model : Any
            Prediction model that exposes a ``predict`` method.
        user_control_columns : dict
            Dictionary specifying columns to control. Keys are "int" or "float"
            and values are ``{column_name: step}`` mappings describing the input
            type and adjustment step for each column.
        reference : pandas.DataFrame
            Reference data used for calculating default values and highlight ranges.
        highlight_columns : list[str], optional
            Columns to highlight based on reference statistics.
        """

        self.model = model
        self.user_control_columns = user_control_columns
        self.reference = reference
        self.highlight_columns = highlight_columns or []

        # Keep columns attribute for compatibility with prediction stage
        int_cols = list(user_control_columns.get("int", {}).keys())
        float_cols = list(user_control_columns.get("float", {}).keys())
        self.columns = int_cols + float_cols
        self.total_df = pd.DataFrame(columns=self.columns)

        # User input widgets generated from provided columns
        self.widgets_dict = {}

        # Buttons
        self.submit_button = widgets.Button(description="Submit", button_style="success")
        self.predict_button = widgets.Button(description="Predict", button_style="info")
        self.submit_button.on_click(self._on_submit)
        self.predict_button.on_click(self._on_predict)

        self.output = widgets.Output()
        self.df_output = widgets.Output()
        self.delete_output = widgets.Output()

        self.delete_buttons = []

        self._build_ui()

    def _build_ui(self):
        widget_list = []
        for dtype, cols in self.user_control_columns.items():
            for col, step in cols.items():
                series = self.reference[col] if col in self.reference else pd.Series(dtype=float)
                mean_val = series.mean() if not series.empty else 0
                min_val = series.min() if not series.empty else 0
                max_val = series.max() if not series.empty else 0

                label_style = 'color:blue; font-weight:bold;' if col in self.highlight_columns else ''
                label = widgets.HTML(
                    value=f"<span style='{label_style}'>{col}</span> ({min_val:.2f} ~ {max_val:.2f})",
                    layout=widgets.Layout(width='500px')
                )

                if dtype == 'int':
                    input_widget = widgets.IntText(value=int(round(mean_val)), step=step)
                else:
                    input_widget = widgets.FloatText(value=round(float(mean_val), 3), step=step)

                self.widgets_dict[col] = input_widget
                widget_list.append(widgets.HBox([label, input_widget]))

        buttons = widgets.HBox([self.submit_button, self.predict_button])
        form = widgets.VBox(widget_list + [buttons, self.delete_output, self.output, self.df_output])
        display(form)

    def _calculate_dependent_columns(self, df):
        # Modular calculation function
        calculations = [
            lambda df: df.assign(장입Total=df[['충진_하단', '충진_중단', '충진_상단']].sum(axis=1))
            # More calculations can be added here
        ]
        for calc_func in calculations:
            df = calc_func(df)
        # Ensure internal column order reflects calculated columns
        self.columns = list(df.columns)
        return df

    def _validate_input(self, df):
        # Separate validation function
        errors = []
        if '장입Total' in df.columns and (df['장입Total'] != 16).any():
            errors.append("'장입Total'은 반드시 16이어야 합니다.")
        return errors

    def _style_dataframe(self):
        # DataFrame styling based on validation
        def highlight_invalid(val):
            return 'background-color: red' if val != 16 else ''

        styled_df = self.total_df.style

        # Highlight 장입Total validity if present
        if '장입Total' in self.total_df.columns:
            styled_df = styled_df.applymap(highlight_invalid, subset=['장입Total'])

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

    def _on_submit(self, b):
        # Gather inputs
        input_data = {col: widget.value for col, widget in self.widgets_dict.items()}
        input_df = pd.DataFrame([input_data], columns=self.columns)

        # Calculate dependent columns
        input_df = self._calculate_dependent_columns(input_df)

        # Validate input
        errors = self._validate_input(input_df)

        # Add to total DataFrame
        self.total_df = pd.concat([self.total_df, input_df], ignore_index=True)

        # Add delete button for the new row
        new_idx = len(self.total_df) - 1
        delete_button = widgets.Button(
            description=f"Delete Row {new_idx}",
            button_style="danger",
            layout=widgets.Layout(width="120px"),
        )

        # Store the index on the button so the callback always has the current
        # value even after other rows are deleted.
        delete_button.idx = new_idx

        def handle_click(btn):
            self._delete_row(btn.idx)

        delete_button.on_click(handle_click)
        self.delete_buttons.append(delete_button)

        self._display_delete_buttons()
        self._style_dataframe()

        with self.output:
            clear_output()
            if errors:
                for err in errors:
                    print(f"Validation Error: {err}")

    def _display_delete_buttons(self):
        buttons_box = widgets.HBox(self.delete_buttons)
        with self.delete_output:
            clear_output()
            display(buttons_box)

    def _delete_row(self, idx):
        # Delete row from DataFrame
        if idx >= len(self.total_df):
            return

        self.total_df = self.total_df.drop(index=idx).reset_index(drop=True)
        del self.delete_buttons[idx]

        # Reset button labels and stored indices without reattaching handlers
        for i, btn in enumerate(self.delete_buttons):
            btn.description = f"Delete Row {i}"
            btn.idx = i

        self._display_delete_buttons()
        self._style_dataframe()

    def _on_predict(self, b):
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
            predictions = self.model.predict(self.total_df[feature_cols])
            result_df = self.total_df.copy()
            result_df['Prediction'] = predictions

            print("예측 결과:")
            display(result_df)

# Example usage (Replace with actual model and reference data)
# xgb_model = trained_xgb_model
# reference_df = pd.read_csv('training_data.csv')
# control_columns = {
#     'int': {'충진_하단': 1},
#     'float': {'충진_중단': 0.5, '충진_상단': 0.1},
# }
# widget = UserControlInputWidget(
#     model=xgb_model,
#     user_control_columns=control_columns,
#     reference=reference_df,
#     highlight_columns=list(control_columns.get('int', {}).keys()) + list(control_columns.get('float', {}).keys()),
# )
