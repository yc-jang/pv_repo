from __future__ import annotations
import os
import joblib
import numpy as np
import pandas as pd
import torch
from typing import Any, Dict, List, Optional, Union, Literal
from sklearn.metrics import r2_score

# TabPFN 라이브러리 및 확장 인터페이스 확인
try:
    from tabpfn import TabPFNRegressor
except ImportError:
    raise ImportError("TabPFN 라이브러리가 필요합니다. 'pip install tabpfn'을 실행하세요.")

# =========================
# TabPFNRunner (v2 Production Ready)
# =========================
class TabPFNRunner(ResultProtocolMixin):
    """TabPFN v2 전용 런너 (Foundation Model 최적화 버전).
    
    CatBoostRunner와 호환성을 유지하면서 ICL(In-Context Learning) 아키텍처를 구현합니다.
    Optuna 최적화가 불필요하므로 제거되었으며, 로컬.ckpt 경로 관리에 집중합니다.
    """

    def __init__(self,
                 model_path: str = "model/tabpfn-v2-regressor.ckpt",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 batch_size: int = 1,
                 ignore_pretraining_limits: bool = False,
                 random_state: int = 42):
        """
        Args:
            model_path: 로컬 체크포인트(.ckpt) 파일의 절대 또는 상대 경로.
            device: 'cuda', 'cpu' 등 연산 장치.
            batch_size: 추론 시 배치 크기 (메모리 상황에 따라 조절).
            ignore_pretraining_limits: True일 경우 10k 샘플/500 피처 제한을 초과하여 작동 시도.
        """
        self.model_name: str = "TabPFN_v2"
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.random_state = random_state

        # 하드웨어 최적화 경고
        if self.device == "cpu":
            # 큰 데이터셋의 경우 CPU 환경에서 환경변수 설정이 필요할 수 있음
            os.environ = "true"
        
        # 모델 초기화 (weights 즉시 로드)
        try:
            self.model = TabPFNRegressor(
                model_path=self.model_path,
                device=self.device,
                batch_size=self.batch_size,
                ignore_pretraining_limits=self.ignore_pretraining_limits,
                random_state=self.random_state
            )
        except Exception as e:
            print(f"Error loading TabPFN weights from {model_path}: {e}")
            self.model = None

        # ResultProtocolMixin 요구 속성
        self.X_train: Optional = None
        self.y_train: Optional = None
        self.X_test: Optional = None
        self.y_test: Optional = None
        self.columns: Optional[pd.Index] = None
        self.ntree_end_: Optional[int] = None  # ICL 모델이므로 항상 None
        self.trained: bool = False

    def _update_data(self,
                     X_train: pd.DataFrame,
                     X_test: pd.DataFrame,
                     y_train: pd.Series,
                     y_test: pd.Series) -> "TabPFNRunner":
        """데이터셋 세팅 및 컬럼 정보 보존."""
        self.X_train = X_train.copy()
        self.X_test = X_test.copy()
        self.y_train = y_train.copy()
        self.y_test = y_test.copy()
        self.columns = X_train.columns
        return self

    def fit(self) -> "TabPFNRunner":
        """
        TabPFN은 가중치 학습이 아닌 데이터 컨텍스트 등록(ICL)을 수행합니다.
        타겟 데이터(y)가 반드시 float 타입이어야 하는 제약사항을 자동 해결합니다.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data is not set. Call _update_data first.")

        # 타겟 데이터 정합성 보장 (회귀 모델은 float 필수)
        y_context = self.y_train.values.astype(float)
        
        # 훈련 데이터셋을 Transformer의 Context Window에 등록
        self.model.fit(self.X_train, y_context)
        
        self.trained = True
        return self

    def predict(self, 
                X: pd.DataFrame, 
                output_type: Literal["mean", "median", "quantiles"] = "mean") -> np.ndarray:
        """
        주어진 데이터에 대해 1회 Forward Pass 추론을 수행합니다.
        
        Args:
            X: 예측할 데이터프레임.
            output_type: 'mean'(기본값), 'median'(MAE 최적화), 'quantiles'(분포 확인).
        """
        if not self.trained:
            raise ValueError("Model is not fitted. Context data must be registered via fit().")
        
        # 훈련 시와 동일한 피처 순서 보장
        if self.columns is not None:
            X = X.reindex(columns=self.columns, copy=False)

        # TabPFN 고유 추론 (Single Forward Pass)
        return self.model.predict(X, output_type=output_type)

    def scores(self) -> Dict[str, float]:
        """테스트셋 기준 주요 성능 지표 반환."""
        if self.X_test is None or self.y_test is None:
            return {}
        
        y_pred = self.predict(self.X_test)
        return {
            "r2": float(r2_score(self.y_test, y_pred)),
            "rmse": _rmse(self.y_test, y_pred)
        }

    def evaluate(self) -> Dict[str, Any]:
        """상세 평가 결과 및 예측 배열 반환."""
        sc = self.scores()
        y_pred = self.predict(self.X_test) if self.X_test is not None else None
        return {**sc, "y_pred": y_pred}

    # ---------- Persistence (Decoupled Load/Save Strategy) ----------
    def save_model(self, 
                   target: str, 
                   filename_pattern: str) -> str:
        """
        모델의 가중치 자체를 저장하는 대신, 훈련 컨텍스트(Data)와 로컬 경로만 저장합니다.
        이는 GPU 환경에서 학습하고 CPU에서 로드할 때 발생하는 충돌을 완벽히 방지합니다.
        """
        if not self.trained:
            raise ValueError("Only trained (fitted) models can be saved.")

        save_dir = "model/tabpfn"
        os.makedirs(save_dir, exist_ok=True)
        
        current_r2 = self.scores().get("r2", 0.0)
        file_name = f"{target}_TabPFN_r2_{current_r2:.4f}_{filename_pattern}.joblib"
        file_path = os.path.join(save_dir, file_name)

        # 저장할 상태 정보 구성
        artifact = {
            "model_type": "TabPFN_v2_Runner",
            "model_path": self.model_path,
            "device": self.device,
            "columns": list(self.columns) if self.columns is not None else None,
            "X_train": self.X_train,
            "y_train": self.y_train,
            "ignore_pretraining_limits": self.ignore_pretraining_limits,
            "random_state": self.random_state,
            "timestamp": _now_iso()
        }
        
        joblib.dump(artifact, file_path)
        return os.path.abspath(file_path)

    def load_model(self, path: str) -> "TabPFNRunner":
        """저장된 아티팩트를 로드하고 현재 환경의 하드웨어에 맞춰 모델을 재구성합니다."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No artifact found at {path}")

        artifact = joblib.load(path)
        
        # 설정 복원
        self.model_path = artifact["model_path"]
        self.columns = pd.Index(artifact["columns"]) if artifact["columns"] else None
        self.X_train = artifact["X_train"]
        self.y_train = artifact["y_train"]
        self.ignore_pretraining_limits = artifact.get("ignore_pretraining_limits", False)
        self.random_state = artifact.get("random_state", 42)

        # 모델 객체 재구성 (현재 인스턴스의 self.device 사용)
        self.model = TabPFNRegressor(
            model_path=self.model_path,
            device=self.device,
            ignore_pretraining_limits=self.ignore_pretraining_limits,
            random_state=self.random_state
        )
        
        # 데이터 컨텍스트 재등록
        if self.X_train is not None and self.y_train is not None:
            self.model.fit(self.X_train, self.y_train.astype(float))
            self.trained = True
            
        return self

    # ---------- Interpretability (Extension Required) ----------
    def get_feature_importance(self, topk: int = 20) -> pd.DataFrame:
        """
        TabPFN은 MDI 기반 중요도가 없으므로 SHAP을 통한 해석을 권장합니다.
        tabpfn-extensions 라이브러리 설치 시 작동합니다.
        """
        try:
            from tabpfn_extensions.interpretability import get_feature_importance
            return get_feature_importance(self.model, self.X_train, self.y_train, topk=topk)
        except ImportError:
            # Fallback: 모델 특성상 uninformative feature에 강하므로 
            # 기본 중요도가 없을 경우 빈 데이터프레임 반환 후 경고
            print("Notice: 'tabpfn-extensions' is required for SHAP-based importance.")
            return pd.DataFrame(columns=["feature", "importance"])
