import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Any
import plotly.graph_objects as go

# ---------------------------------------------------------
# [0] 환경 설정 및 로거 초기화
# ---------------------------------------------------------
# 로거 설정 (이미 설정되어 있다면 생략 가능)
logger = logging.getLogger(__name__)

# (예시) 가상의 Runner 클래스 (실제 환경에 맞게 교체하세요)
class CatBoostRunner:
    def load_model(self, path: str):
        # logger.debug(f"Loading model from {path}")
        return self # Chainable
    
    def predict(self, inputs):
        # Mock prediction: 데모용 랜덤 값 생성
        return np.random.uniform(50, 100, size=len(inputs))

def run_prediction_and_plot_overlay(
    model_dir: Path, 
    model_input: pd.DataFrame, 
    full_to_short: Dict[str, str]
):
    """
    1. 모델 파일을 찾아 예측을 수행하고
    2. 결과를 하나의 Plotly 그래프에 겹쳐서 시각화합니다.
    """
    logger.info("="*60)
    logger.info("모델 예측 및 통합 시각화 프로세스를 시작합니다.")

    # -----------------------------------------------------
    # [1] 모델 파일 탐색 및 예측 (Prediction Phase)
    # -----------------------------------------------------
    # 실제 환경에서는 glob 패턴이나 find_models 함수 사용
    paths = list(model_dir.glob("*.cbm")) 
    
    valid_targets: Set[str] = set(full_to_short.values())
    prediction_results = {} # {target_name: prediction_array}

    if not paths:
        logger.warning(f"경로에 모델 파일이 없습니다: {model_dir}")
        # (테스트를 위해 빈 딕셔너리로 진행하지 않고 리턴)
        return

    for p in paths:
        filename = p.name
        
        # 1-1. 타겟 이름 파싱 및 검증
        try:
            if "_CAT" in filename:
                target_candidate = filename.split("_CAT")[0]
            else:
                logger.debug(f"Skip: '_CAT' 키워드가 없는 파일입니다. ({filename})")
                continue

            if target_candidate in valid_targets:
                target = target_candidate
            else:
                logger.warning(f"Skip: 유효하지 않은 타겟입니다. ({target_candidate})")
                continue

        except Exception as e:
            logger.error(f"파일명 파싱 중 에러 발생 ({filename}): {e}")
            continue

        # 1-2. 모델 로딩 및 예측
        try:
            logger.info(f"모델 로딩 중... Target: {target}")
            
            # 실제 모델 로드 및 예측
            runner = CatBoostRunner().load_model(str(p))
            model_predicted = runner.predict(model_input) # 결과: np.array
            
            prediction_results[target] = model_predicted
            
            logger.debug(f"예측 완료. Target: {target}, Shape: {model_predicted.shape}")

        except Exception as e:
            logger.error(f"예측 수행 중 치명적 오류 발생 (Target: {target}): {e}")
            continue

    if not prediction_results:
        logger.error("유효한 예측 결과가 없습니다. 시각화를 종료합니다.")
        return

    # -----------------------------------------------------
    # [2] 시각화 (Visualization Phase) - Overlay Graph
    # -----------------------------------------------------
    logger.info(f"총 {len(prediction_results)}개 타겟에 대한 통합 시각화를 시작합니다.")
    
    # 2-1. X축 라벨 가공 (핵심 요구사항)
    # 첫 번째(원본)는 인덱스 그대로(LOT번호), 그 외는 뒤의 버전(v1, v2..)만 사용
    raw_indices = model_input.index.tolist()
    x_labels = []
    
    for i, idx in enumerate(raw_indices):
        if i == 0:
            # 원본 (예: LOT01)
            x_labels.append(str(idx))
        else:
            # 시뮬레이션 (예: LOT01_v1 -> v1)
            parts = str(idx).split('_')
            # 만약 '_'가 없다면 그대로 사용, 있다면 뒤쪽(v1) 사용
            label = parts[-1] if len(parts) > 1 else str(idx)
            x_labels.append(label)

    # 2-2. 단일 Figure 생성
    fig = go.Figure()

    # 2-3. 그래프 그리기 (Loop)
    # 색상을 자동으로 구분하기 위해 Plotly가 알아서 할당하지만,
    # 필요하다면 color list를 순회하며 지정할 수도 있습니다.
    for target_name, pred_values in prediction_results.items():
        
        # Line + Marker 차트 추가
        fig.add_trace(
            go.Scatter(
                x=x_labels, 
                y=pred_values,
                mode='lines+markers+text', # 선+점+값 텍스트 표시
                name=target_name,          # 범례에 표시될 이름
                text=[f"{v:.1f}" for v in pred_values], # 값 표시 (소수점 1자리)
                textposition="top center",
                line=dict(width=2),
                marker=dict(size=6)
            )
        )

    # 2-4. 레이아웃 및 범례(Legend) 설정 (요구사항 완벽 반영)
    fig.update_layout(
        template='plotly_white', # 깔끔한 흰색 배경
        title_text="<b>Model Simulation Results (Overlay)</b>",
        title_x=0.5,
        height=600,              # 하나의 큰 그래프이므로 높이 고정
        
        # [핵심] 범례 설정: 그래프 상단 오른쪽에 가로로 배치
        legend=dict(
            orientation='h',      # 가로 배치
            yanchor='bottom', 
            y=1.02,               # 그래프 상단 위쪽 (겹치지 않게)
            xanchor='right', 
            x=1                   # 오른쪽 정렬
        ),
        
        # 마우스 오버 시 X축 기준으로 모든 타겟의 값을 한 번에 비교
        hovermode="x unified"     
    )

    # 2-5. 축 설정
    fig.update_xaxes(title_text="Simulation Version (Base -> Variations)")
    fig.update_yaxes(title_text="Predicted Value")

    logger.info("통합 시각화 그래프 생성 완료.")
    fig.show()

# ---------------------------------------------------------
# [실행 테스트]
# ---------------------------------------------------------
if __name__ == "__main__":
    # 로깅 레벨 설정 (Info 이상 출력)
    logging.basicConfig(level=logging.INFO)

    # 1. 임시 데이터 생성 (LOT01 + v1 + v2 + v3)
    input_data = pd.DataFrame(
        np.random.rand(4, 5), 
        columns=[f"Feat_{i}" for i in range(5)],
        index=["LOT01", "LOT01_v1", "LOT01_v2", "LOT01_v3"]
    )

    # 2. 매핑 정보 (파일명의 Prefix -> 실제 타겟 이름)
    mapping = {
        "TargetA": "TargetA", 
        "TargetB": "TargetB",
        "TargetC": "TargetC"
    }
    
    # 3. 모델 디렉토리 (테스트용)
    model_dir_path = Path("./models") 
    
    # 실행
    run_prediction_and_plot_overlay(model_dir_path, input_data, mapping)
