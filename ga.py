import numpy as np
import pandas as pd
import pygad
import xgboost as xgb
import catboost as cb
from typing import Dict, Any, Tuple, Union, List

class GeneticOptimizer:
    def __init__(self, model, model_type: str, feature_names: list, X_train: pd.DataFrame, scaler: Any, desired_value: Union[float, List[float]], important_feature: list, optimize_mode: str = 'single'):
        self.model = model
        self.model_type = model_type
        self.feature_names = feature_names
        self.X_train = X_train[self.feature_names].copy()
        self.scaler = scaler
        self.desired_value = desired_value
        self.important_features = important_feature
        self.feature_bounds = None
        self.optimize_mode = optimize_mode
        
        if self.important_features:
            assert all(f in self.feature_names for f in self.important_features), "important_features must be subset of feature_names"
        assert self.X_train.shape[1] == len(self.feature_names), "X_train columns must match feature_names"
        assert self.optimize_mode in ['single', 'all'], "optimize_mode must be 'single' or 'all'"
        print("X_train shape:", self.X_train.shape)
        print("Optimize mode:", self.optimize_mode)

    def set_feature_bounds(self, X: pd.DataFrame) -> None:
        self.feature_bounds = {
            f: (X[f].min(), X[f].max()) for f in self.feature_names
        }
        if self.important_features:
            self.feature_bounds = {f: self.feature_bounds[f] for f in self.important_features}
        print("Feature bounds:", self.feature_bounds)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.feature_names]
        if self.scaler:
            X = pd.DataFrame(self.scaler.transform(X), columns=self.feature_names)
        if self.model_type == 'xgboost':
            dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
            return self.model.predict(dmatrix)
        elif self.model_type == 'catboost':
            return self.model.predict(X)
        else:
            raise ValueError("model_type must be 'xgboost' or 'catboost'")

    def fitness_function(self, X: pd.DataFrame, **params: float) -> float:
        X = X[self.feature_names].copy()
        X_origin = X.copy()
        
        for f in self.important_features:
            X[f] = params[f]
        
        if self.scaler:
            X = pd.DataFrame(self.scaler.transform(X), columns=self.feature_names)
            X_origin = pd.DataFrame(self.scaler.transform(X_origin), columns=self.feature_names)
        
        y_pred = self.predict(X)
        origin_pred = self.predict(X_origin)
        
        # desired_value 처리
        desired = np.array([self.desired_value] * len(X) if isinstance(self.desired_value, (int, float)) else self.desired_value)
        assert len(desired) == len(X), "desired_value list must match X rows"
        
        # desired_value 제약
        if not all(desired - 10 <= y_pred) or not all(y_pred <= desired + 10):
            return -1e10
        
        base_score = np.mean((y_pred - desired) ** 2)
        penalty_weight = 1000
        penalty = penalty_weight * np.mean(np.maximum(0, origin_pred - y_pred))
        return -base_score - penalty

    def optimize(self, for_optimal: pd.DataFrame, init_point: int = 5, n_iter: int = 20) -> Tuple[pd.DataFrame, float]:
        self.set_feature_bounds(self.X_train)
        
        for_optimal = for_optimal[self.feature_names].copy()
        if self.optimize_mode == 'single':
            assert for_optimal.shape[0] == 1, "Single mode requires exactly one row"
        # 'all' 모드는 다중 행 허용, 별도 assert 없음
        
        def obj_func(ga_instance: pygad.GA, solution: np.ndarray, solution_idx: int) -> float:
            params = {f: solution[i] for i, f in enumerate(self.important_features)}
            return self.fitness_function(for_optimal, **params)
        
        ga_instance = pygad.GA(
            num_generations=n_iter,
            num_parents_mating=init_point,
            fitness_func=obj_func,
            sol_per_pop=50,
            num_genes=len(self.important_features),
            init_range_low=[self.feature_bounds[f][0] for f in self.important_features],
            init_range_high=[self.feature_bounds[f][1] for f in self.important_features],
            mutation_probability=0.05,
            crossover_probability=0.8,
            random_seed=42
        )
        
        ga_instance.run()
        
        optimal_solution, optimal_fitness, optimal_solution_idx = ga_instance.best_solution()
        print("Optimal solution:", optimal_solution)
        
        optimal_input = for_optimal.copy()
        for i, f in enumerate(self.important_features):
            optimal_input[f] = optimal_solution[i]
        optimal_input = optimal_input[self.feature_names]
        
        print("Optimal input:", optimal_input)
        
        return optimal_input, optimal_fitness

# 사용 예시
feature_names = [f'f{i}' for i in range(102)]
end2end_data = pd.DataFrame(np.random.uniform(0.1, 1.0, (6125, 102)), columns=feature_names)
y = np.random.uniform(100, 300, 6125)

end2end_data = end2end_data[feature_names]
X_train, X_test, y_train, y_test = train_test_split(end2end_data, y, test_size=0.2, random_state=42)

def train_cat_with_r2():
    model = cb.CatBoostRegressor(
        iterations=1000,
        depth=6,
        learning_rate=0.05,
        random_seed=42,
        verbose=0
    )
    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print("CatBoost R²:", r2)
    return model

cat_model = train_cat_with_r2()

importance_features = [f'f{i}' for i in range(12)]

# 'single' 모드 예시 (단일 행)
cat_optimizer_single = GeneticOptimizer(
    model=cat_model,
    model_type='catboost',
    feature_names=feature_names,
    X_train=X_train,
    scaler=None,
    desired_value=215,
    important_feature=importance_features,
    optimize_mode='single'
)
optimal_input, optimal_value = cat_optimizer_single.optimize(for_optimal=X_test.iloc[[0]], init_point=5, n_iter=20)
print("\nSingle Mode Results:")
print("Optimal input:", optimal_input)
print("Optimal value:", optimal_value)

# 'all' 모드 예시 (다중 행)
cat_optimizer_all = GeneticOptimizer(
    model=cat_model,
    model_type='catboost',
    feature_names=feature_names,
    X_train=X_train,
    scaler=None,
    desired_value=[215] * len(X_test),  # 다중 행에 대한 desired_value 리스트
    important_feature=importance_features,
    optimize_mode='all'
)
optimal_input_all, optimal_value_all = cat_optimizer_all.optimize(for_optimal=X_test, init_point=5, n_iter=20)
print("\nAll Mode Results:")
print("Optimal input:", optimal_input_all)
print("Optimal value:", optimal_value_all)
