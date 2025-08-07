import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import catboost as cb
from bayes_opt import BayesianOptimization
import shap
import dice_ml
from sklearn.metrics import r2_score

class TargetOptimizer:
    def __init__(self, model, model_type, feature_names, scaler, desired_range=[100, 110], important_features=None):
        """
        BayesianOptimization으로 타겟 성능을 최적화하는 클래스.

        Parameters:
        - model: 학습된 모델 (XGBoost 또는 CatBoost).
        - model_type: 'xgboost' 또는 'catboost'.
        - feature_names: 피처 이름 리스트.
        - scaler: 전처리 객체 (예: StandardScaler).
        - desired_range: 타겟 예측값 범위 [min, max].
        - important_features: 우선 선택할 피처 리스트 (기본: None, SHAP 사용).
        """
        self.model = model
        self.model_type = model_type
        self.feature_names = feature_names
        self.scaler = scaler
        self.desired_range = desired_range
        self.important_features = important_features
        self.target_mid = sum(desired_range) / 2  # desired_range 중앙값
        self.feature_bounds = None

    def set_feature_bounds(self, X_scaled):
        """피처 값 범위 설정 (스케일링된 데이터 기준)."""
        self.feature_bounds = {
            f: (X_scaled[f].min(), X_scaled[f].max()) for f in self.feature_names
        }
        if self.important_features:
            self.feature_bounds = {f: self.feature_bounds[f] for f in self.important_features}

    def select_important_features(self, X_scaled, n_features=20):
        """SHAP으로 중요 피처 선택."""
        if self.important_features:
            return self.important_features
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_scaled)
        shap_importance = np.abs(shap_values).mean(axis=0)
        indices = np.argsort(-shap_importance)[:n_features]
        return [self.feature_names[i] for i in indices]

    def objective_function(self, **params):
        """최적화 목적 함수: 예측값을 desired_range에 맞춤."""
        X = pd.DataFrame([params[f] for f in self.feature_names], index=self.feature_names).T
        X = X[self.feature_names]  # 열 순서 고정
        if self.scaler:
            X = pd.DataFrame(self.scaler.transform(X), columns=self.feature_names)
        
        if self.model_type == 'xgboost':
            dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
            y_pred = self.model.predict(dmatrix)[0]
        elif self.model_type == 'catboost':
            y_pred = self.model.predict(X)[0]
        else:
            raise ValueError("model_type must be 'xgboost' or 'catboost'")
        
        return -(y_pred - self.target_mid) ** 2  # 최소화 문제

    def optimize(self, X_scaled, init_points=5, n_iter=20):
        """BayesianOptimization 실행."""
        self.set_feature_bounds(X_scaled)
        if not self.important_features:
            self.important_features = self.select_important_features(X_scaled)
        
        bo = BayesianOptimization(
            f=self.objective_function,
            pbounds={f: self.feature_bounds[f] for f in self.important_features},
            random_state=42,
            verbose=2
        )
        bo.maximize(init_points=init_points, n_iter=n_iter)
        
        # 최적 입력 생성 (모든 피처 포함)
        optimal_params = bo.max['params']
        optimal_input = pd.DataFrame(np.zeros((1, len(self.feature_names))), columns=self.feature_names)
        for f in self.important_features:
            optimal_input[f] = optimal_params[f]
        
        # 스케일링 적용
        if self.scaler:
            optimal_input = pd.DataFrame(self.scaler.transform(optimal_input), columns=self.feature_names)
        
        return optimal_input, bo.max['target']

    def predict(self, X):
        """최적화된 입력으로 예측."""
        X = X[self.feature_names]
        if self.model_type == 'xgboost':
            dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
            return self.model.predict(dmatrix)[0]
        elif self.model_type == 'catboost':
            return self.model.predict(X)[0]

# 사용 예시
# 데이터 준비
feature_names = [f'f{i}' for i in range(102)]
end2end_data = pd.DataFrame(np.random.rand(6125, 102), columns=feature_names)
y = np.random.rand(6125) * 100  # desired_range=[100, 110]

# 열 순서 고정
end2end_data = end2end_data[feature_names]
X_train, X_test, y_train, y_test = train_test_split(end2end_data, y, test_size=0.2, random_state=42)

# 전처리
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names)

# XGBoost 모델 학습
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}
dtrain = xgb.DMatrix(X_train_scaled, label=y_train, feature_names=feature_names)
dvalid = xgb.DMatrix(X_test_scaled, label=y_test, feature_names=feature_names)
xgb_model = xgb.train(
    xgb_params, dtrain, num_boost_round=1000,
    evals=[(dtrain, 'train'), (dvalid, 'valid')],
    early_stopping_rounds=50, verbose_eval=False
)

# CatBoost 모델 학습
cat_model = cb.CatBoostRegressor(
    iterations=1000,
    depth=6,
    learning_rate=0.05,
    random_seed=42,
    verbose=0
)
cat_model.fit(X_train_scaled, y_train, eval_set=(X_test_scaled, y_test), early_stopping_rounds=50)

# 특정 피처 우선 선택 (예: 상위 20개 피처)
important_features = [f'f{i}' for i in range(20)]  # 사용자 지정 예시

# TargetOptimizer 사용
xgb_optimizer = TargetOptimizer(
    model=xgb_model,
    model_type='xgboost',
    feature_names=feature_names,
    scaler=scaler,
    desired_range=[100, 110],
    important_features=important_features
)
xgb_optimal_input, xgb_optimal_value = xgb_optimizer.optimize(X_train_scaled, init_points=5, n_iter=20)
xgb_pred = xgb_optimizer.predict(xgb_optimal_input)
print("XGBoost optimal input:", xgb_optimal_input)
print("XGBoost predicted value:", xgb_pred)

cat_optimizer = TargetOptimizer(
    model=cat_model,
    model_type='catboost',
    feature_names=feature_names,
    scaler=scaler,
    desired_range=[100, 110],
    important_features=important_features
)
cat_optimal_input, cat_optimal_value = cat_optimizer.optimize(X_train_scaled, init_points=5, n_iter=20)
cat_pred = cat_optimizer.predict(cat_optimal_input)
print("CatBoost optimal input:", cat_optimal_input)
print("CatBoost predicted value:", cat_pred)

# DiCE-ML 통합
data_df = pd.concat([X_train_scaled, pd.Series(y_train, name='target')], axis=1)
d = dice_ml.Data(
    dataframe=data_df,
    continuous_features=feature_names,
    outcome_name='target'
)

# XGBoost predict_fn
def xgb_predict_fn(X, data_interface=None):
    X = pd.DataFrame(X, columns=feature_names)[feature_names]
    X_transformed = pd.DataFrame(scaler.transform(X), columns=feature_names)
    dmatrix = xgb.DMatrix(X_transformed, feature_names=feature_names)
    return xgb_model.predict(dmatrix)

# CatBoost predict_fn
def cat_predict_fn(X, data_interface=None):
    X = pd.DataFrame(X, columns=feature_names)[feature_names]
    X_transformed = pd.DataFrame(scaler.transform(X), columns=feature_names)
    return cat_model.predict(X_transformed)

# DiCE-ML 모델 설정
xgb_dice_model = dice_ml.Model(
    model=xgb_model,
    backend={
        'model': 'xgboost_model.XGBoostModel',
        'explainer': 'dice_xgboost.DiceXGBoost'
    },
    model_type='regressor',
    func=xgb_predict_fn
)
cat_dice_model = dice_ml.Model(
    model=cat_model,
    backend='sklearn',
    model_type='regressor'
)

# 카운터팩추얼 생성
xgb_exp = dice_ml.Dice(d, xgb_dice_model, method='genetic')
xgb_dice_exp = xgb_exp.generate_counterfactuals(
    query_instance=xgb_optimal_input,
    total_CFs=10,
    desired_range=[100, 110],
    features_to_vary=important_features
)
xgb_dice_exp.visualize_as_dataframe(show_only_changes=True)

cat_exp = dice_ml.Dice(d, cat_dice_model, method='genetic')
cat_dice_exp = cat_exp.generate_counterfactuals(
    query_instance=cat_optimal_input,
    total_CFs=10,
    desired_range=[100, 110],
    features_to_vary=important_features
)
cat_dice_exp.visualize_as_dataframe(show_only_changes=True)
