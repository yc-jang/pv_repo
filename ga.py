import numpy as np
import pandas as pd
import pygad
import xgboost as xgb
import catboost as cb
from typing import Dict, Any, Tuple, Union, List

class GeneticOptimizer:
    def __init__(self, model, model_type: str, feature_names: list, X_train: pd.DataFrame, scaler: Any, desired_value: Union[float, List[float]], important_feature: list):
        self.model = model
        self.model_type = model_type
        self.feature_names = feature_names
        self.X_train = X_train[self.feature_names].copy()
        self.scaler = scaler
        self.desired_value = desired_value
        self.important_features = important_feature
        self.feature_bounds = None
        
        if self.important_features:
            assert all(f in self.feature_names for f in self.important_features), "important_features must be subset of feature_names"
        assert self.X_train.shape[1] == len(self.feature_names), "X_train columns must match feature_names"
        print("X_train shape:", self.X_train.shape)

    def set_feature_bounds(self, X: pd.DataFrame) -> None:
        """피처 값 범위 설정 (원본 데이터 기준)."""
        self.feature_bounds = {
            f: (X[f].min(), X[f].max()) for f in self.feature_names
        }
        if self.important_features:
            self.feature_bounds = {f: self.feature_bounds[f] for f in self.important_features}
        print("Feature bounds:", self.feature_bounds)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """다중 행 예측 지원."""
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
        """GA fitness 함수: important_features만 조정, 나머지는 X 유지.
        
        Args:
            X (pd.DataFrame): 입력 데이터 (for_optimal).
            **params (float): important_features에 해당하는 피처 값들.

        Returns:
            float: -base_score - penalty, 제약 위반 시 -1e10.
        """
        X = X[self.feature_names].copy()  # 순서 고정
        X_origin = X.copy()  # 원래 모델 입력 복사
        
        for f in self.important_features:
            X[f] = params[f]  # important_features만 업데이트
        
        if self.scaler:
            X = pd.DataFrame(self.scaler.transform(X), columns=self.feature_names)  # 스케일링 적용
            X_origin = pd.DataFrame(self.scaler.transform(X_origin), columns=self.feature_names)  # X_origin에도 스케일링
        
        if self.model_type == 'xgboost':
            dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
            dmatrix_origin = xgb.DMatrix(X_origin, feature_names=self.feature_names)
            y_pred = self.model.predict(dmatrix)[0]
            origin_pred = self.model.predict(dmatrix_origin)[0]
        elif self.model_type == 'catboost':
            y_pred = self.model.predict(X)[0]
            origin_pred = self.model.predict(X_origin)[0]
        else:
            raise ValueError("model_type must be 'xgboost' or 'catboost'")
        
        # desired_value 제약
        if not (self.desired_value - 10 <= y_pred <= self.desired_value + 10):
            return -1e10  # 범위 벗어나면 큰 패널티
        
        base_score = (y_pred - self.desired_value) ** 2
        penalty_weight = 1000
        penalty = penalty_weight * max(0, origin_pred - y_pred)
        return -base_score - penalty  # -base_score - penalty 반환

    def optimize(self, for_optimal: pd.DataFrame, init_point: int = 5, n_iter: int = 20) -> Tuple[pd.DataFrame, float]:
        """유전 알고리즘으로 최적화 실행 (단일 인덱스 지원).
        
        Args:
            for_optimal (pd.DataFrame): 최적화 대상 입력 데이터 (단일 행).
            init_point (int): 부모 수 (num_parents_mating).
            n_iter (int): 세대 수 (num_generations).

        Returns:
            Tuple[pd.DataFrame, float]: 최적 입력과 최적 fitness 값.
        """
        self.set_feature_bounds(self.X_train)  # 피처 범위 설정
        
        optimal_input = for_optimal[self.feature_names].copy()  # 최적화 대상 입력 복사
        assert optimal_input.shape[0] == 1, "for_optimal must contain exactly one row"  # 단일 행 확인
        
        def obj_func(ga_instance: pygad.GA, solution: np.ndarray, solution_idx: int) -> float:
            params = {f: solution[i] for i, f in enumerate(self.important_features)}
            return self.fitness_function(optimal_input, **params)
        
        # PyGAD 설정
        ga_instance = pygad.GA(
            num_generations=n_iter,  # 세대 수
            num_parents_mating=init_point,  # 부모 수
            fitness_func=obj_func,  # fitness 함수
            sol_per_pop=50,  # 인구 크기
            num_genes=len(self.important_features),  # 유전자 수
            init_range_low=[self.feature_bounds[f][0] for f in self.important_features],
            init_range_high=[self.feature_bounds[f][1] for f in self.important_features],
            mutation_probability=0.05,  # 돌연변이 확률
            crossover_probability=0.8,  # 교차 확률
            random_seed=42  # 재현성
        )
        
        ga_instance.run()  # GA 실행
        
        # 최적 솔루션 추출
        optimal_solution, optimal_fitness, optimal_solution_idx = ga_instance.best_solution()
        print("Optimal solution:", optimal_solution)  # 디버깅: 최적 솔루션 출력
        
        # 최적 입력 생성
        optimal_input = optimal_input.copy()
        for i, f in enumerate(self.important_features):
            optimal_input[f] = optimal_solution[i]
        optimal_input = optimal_input[self.feature_names]
        
        print("Optimal input:", optimal_input)  # 디버깅: 최적 입력 출력
        
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
cat_optimizer = GeneticOptimizer(
    model=cat_model,
    model_type='catboost',
    feature_names=feature_names,
    X_train=X_train,
    scaler=None,
    desired_value=215,
    important_feature=importance_features
)

# 단일 인덱스 테스트
optimal_input, optimal_value = cat_optimizer.optimize(for_optimal=X_test.iloc[[0]], init_point=5, n_iter=20)
print("\nFinal Results (index 0):")
print("Optimal input:", optimal_input)
print("Optimal value:", optimal_value)
