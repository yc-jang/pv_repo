import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

def export_report_to_excel(
    optimization_result: Dict[str, Any],
    base_input_df: pd.DataFrame,
    output_paths: Dict[str, Path],
    req_info: Dict[str, str]  # {req_number, req_date, semi_prod_code} 묶음 권장
) -> Dict[str, Path]:
    """
    최적화 결과(MOO)를 분석하여 엑셀 레포트(품질 예측, 공정 조건, 재료 조건)로 저장합니다.
    """
    
    # -------------------------------------------------------------------------
    # [1] Guard Clauses: 데이터 유효성 검증 (Fail Fast)
    # -------------------------------------------------------------------------
    results = optimization_result.get("results", {})
    artifacts = optimization_result.get("artifacts")

    if not results:
        logger.warning("⚠️ 최적화 결과(results)가 비어있어 레포트를 생성할 수 없습니다.")
        return output_paths
    
    if not artifacts:
        logger.error("❌ Artifacts 정보가 누락되었습니다.")
        return output_paths

    # -------------------------------------------------------------------------
    # [2] 설정 및 메타데이터 준비
    # -------------------------------------------------------------------------
    optim_mode = optimization_result.get("usedcol_for_optim", "unknown")
    req_number = req_info.get("req_number", "UNKNOWN")
    
    # 모델 매핑 (Artifacts에서 모델 객체 로드)
    # 가정: suffix가 'cat'이면 cat_models, 'tab'이면 tabpfn_models를 사용
    model_repo = {
        "cat": getattr(artifacts, "cat_models", {}),
        "tab": getattr(artifacts, "tabpfn_models", {})
    }
    
    feature_names: List[str] = artifacts.feature_names
    important_features: List[str] = artifacts.important_features
    
    # (중요) 컬럼 분류 정보가 artifacts에 있다고 가정하거나, 별도 관리 필요
    # 여기서는 artifacts에서 가져오는 것으로 가정
    cond_cols = getattr(artifacts, "condition_columns", []) 
    matr_cols = getattr(artifacts, "material_columns", [])

    # -------------------------------------------------------------------------
    # [3] 예측 리포트 생성 (Looping & Prediction)
    # -------------------------------------------------------------------------
    quality_predictions: List[pd.DataFrame] = []
    final_decision_values = None  # 최종 제어값 저장용

    for result_key, mo_res in results.items():
        # Key 파싱 (예: "Result_cat" -> "cat")
        model_type = result_key.split("_")[-1]
        target_models = model_repo.get(model_type)

        if not target_models:
            logger.debug(f"Skip: '{model_type}'에 해당하는 모델 저장소가 없습니다.")
            continue

        # 3-1. Pareto Front에서 최적해 선정
        # Knee Point 등을 활용하여 최적의 1행 추출
        pareto_df = mo_res.pareto_df
        if pareto_df.empty:
            continue
            
        top_row, _ = select_best_and_knee(pareto_df)
        
        # 3-2. 의사결정 변수(Decision) 적용하여 시뮬레이션 입력값 생성
        decision = _pick_decision_from_pareto(top_row, important_features)
        
        # X_before: 최적화 전, X_after: 최적화 후
        X_before, X_after = _apply_decision(base_input_df, decision, feature_names)
        
        # (나중을 위해 마지막 유효한 X_after를 제어값으로 저장)
        final_decision_values = X_after

        # 3-3. 타겟별 예측 수행
        # target_models가 {target_name: model_obj} 형태라고 가정
        for target_name, model_obj in target_models.items():
            try:
                pred_before = _predict_all(model_obj, X_before, feature_names)
                pred_after = _predict_all(model_obj, X_after, feature_names)

                # 결과 행 생성
                row_df = pd.DataFrame({
                    "요청번호": [req_number],
                    "Target": [target_name],
                    "Model": [model_type],
                    "Before": [pred_before],
                    "After": [pred_after],
                    "Diff": [pred_after - pred_before]
                })
                quality_predictions.append(row_df)

            except Exception as e:
                logger.error(f"예측 실패 ({model_type}-{target_name}): {e}")

    # -------------------------------------------------------------------------
    # [4] 결과 집계 및 데이터프레임 분리
    # -------------------------------------------------------------------------
    if not quality_predictions:
        logger.warning("생성된 예측 결과가 없습니다.")
        return output_paths

    # 4-1. 품질 예측 리포트 (QUAL)
    df_qual = pd.concat(quality_predictions, ignore_index=True).set_index("요청번호")

    # 4-2. 제어 인자 리포트 (COND / MATR)
    if final_decision_values is not None:
        # 중요 인자값만 추출
        control_values = final_decision_values[important_features].iloc[[0]].copy()
        
        # (A) 공정 조건 (COND)
        # cond_cols가 정의되지 않았을 경우를 대비해 교집합 사용
        valid_cond_cols = [c for c in cond_cols if c in control_values.columns]
        df_cond = control_values[valid_cond_cols].reset_index(drop=True)
        df_cond["요청번호"] = req_number
        df_cond = df_cond.set_index("요청번호")

        # (B) 재료 조건 (MATR) - 옵션에 따라 저장
        df_matr = None
        if optim_mode == "matr_cond":
            valid_matr_cols = [c for c in matr_cols if c in control_values.columns]
            df_matr = control_values[valid_matr_cols].reset_index(drop=True)
            df_matr["요청번호"] = req_number
            df_matr = df_matr.set_index("요청번호")
    else:
        logger.warning("최적화된 제어 인자(X_after)를 찾을 수 없습니다.")
        df_cond, df_matr = pd.DataFrame(), None

    # -------------------------------------------------------------------------
    # [5] 엑셀 저장 (Export)
    # -------------------------------------------------------------------------
    # 저장할 대상들을 딕셔너리로 관리하여 반복문으로 깔끔하게 처리
    export_targets = {
        "QUAL": df_qual,
        "COND": df_cond,
        "MATR": df_matr
    }

    for key, df in export_targets.items():
        # 데이터프레임이 존재하고(None 아님), 비어있지 않으며, 경로가 정의된 경우
        if df is not None and not df.empty and key in output_paths:
            try:
                _to_excel(df, output_paths[key])
                logger.info(f"✅ [{key}] 리포트 저장 완료: {output_paths[key].name}")
            except Exception as e:
                logger.error(f"❌ [{key}] 저장 실패: {e}")

    return output_paths
