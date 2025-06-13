import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output

class UserControlWidget:
    def __init__(self, model, columns):
        self.model = model
        self.columns = columns
        self.total_df = pd.DataFrame(columns=columns)

        # User input widgets
        self.input_widgets = {
            '충진_하단': widgets.FloatText(description='충진_하단'),
            '충진_중단': widgets.FloatText(description='충진_중단'),
            '충진_상단': widgets.FloatText(description='충진_상단'),
        }

        # Buttons
        self.submit_button = widgets.Button(description='Submit', button_style='success')
        self.predict_button = widgets.Button(description='Predict', button_style='info')
        self.submit_button.on_click(self._on_submit)
        self.predict_button.on_click(self._on_predict)

        self.output = widgets.Output()
        self.df_output = widgets.Output()

        self.delete_buttons = []

        self._display_widgets()

    def _display_widgets(self):
        inputs = widgets.VBox(list(self.input_widgets.values()))
        buttons = widgets.HBox([self.submit_button, self.predict_button])
        display(inputs, buttons, self.output, self.df_output)

    def _calculate_dependent_columns(self, df):
        # Modular calculation function
        calculations = [
            lambda df: df.assign(장입Total=df[['충진_하단', '충진_중단', '충진_상단']].sum(axis=1))
            # More calculations can be added here
        ]
        for calc_func in calculations:
            df = calc_func(df)
        return df

    def _validate_input(self, df):
        # Separate validation function
        errors = []
        if (df['장입Total'] != 16).any():
            errors.append("'장입Total'은 반드시 16이어야 합니다.")
        return errors

    def _style_dataframe(self):
        # DataFrame styling based on validation
        def highlight_invalid(val):
            return 'background-color: red' if val != 16 else ''

        styled_df = self.total_df.style.applymap(
            highlight_invalid, subset=['장입Total']
        )
        with self.df_output:
            clear_output()
            display(styled_df)

    def _on_submit(self, b):
        # Gather inputs
        input_data = {col: widget.value for col, widget in self.input_widgets.items()}
        input_df = pd.DataFrame([input_data], columns=self.columns)

        # Calculate dependent columns
        input_df = self._calculate_dependent_columns(input_df)

        # Validate input
        errors = self._validate_input(input_df)

        # Add to total DataFrame
        self.total_df = pd.concat([self.total_df, input_df], ignore_index=True)

        # Add delete button
        delete_button = widgets.Button(description=f"Delete Row {len(self.total_df)-1}", button_style='danger', layout=widgets.Layout(width='120px'))
        delete_button.on_click(lambda btn, idx=len(self.total_df)-1: self._delete_row(idx))
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
        with self.output:
            clear_output()
            display(buttons_box)

    def _delete_row(self, idx):
        # Delete row from DataFrame
        self.total_df = self.total_df.drop(index=idx).reset_index(drop=True)
        del self.delete_buttons[idx]

        # Reset button labels and handlers
        for i, btn in enumerate(self.delete_buttons):
            btn.description = f"Delete Row {i}"
            btn.on_click(lambda btn, idx=i: self._delete_row(idx))

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

            predictions = self.model.predict(self.total_df[self.columns])
            result_df = self.total_df.copy()
            result_df['Prediction'] = predictions

            print("예측 결과:")
            display(result_df)

# Example usage (Replace with actual model)
# xgb_model = trained_xgb_model
data_columns = ['충진_하단', '충진_중단', '충진_상단', '장입Total']
# widget = UserControlWidget(model=xgb_model, columns=data_columns)
