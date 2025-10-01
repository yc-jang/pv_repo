from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

@staticmethod
def _collect_pareto(res, problem: "BatchMultiObjectiveProblem") -> pd.DataFrame:
    """항상 '최종 1해'만 반환.
    - feasible Pareto(res.opt)가 있으면: ∑F 최소 1개만
    - 없으면: 마지막 pop에서 Y=[CV,F...] 기반 NDS(front-0) → (CV, ∑F) 오름차순으로 1개
    반환 스키마: x::*, f::*, g::* (+ note)
    """

    # ---------- 유틸: 컬럼명 ----------
    def _names(prob: "BatchMultiObjectiveProblem") -> Tuple[List[str], List[str], List[str]]:
        x_cols = [f"x::{f}" for f in prob.important_features]
        f_cols = [f"f::{t}" for t in prob.targets]
        g_names: List[str] = []
        for t in prob.targets:
            sp = prob.specs[t]
            if sp["lower"] is not None:
                g_names.append(f"{t}_lower")
            if sp["upper"] is not None:
                g_names.append(f"{t}_upper")
        g_cols = [f"g::{name}" for name in g_names] if g_names else []
        return x_cols, f_cols, g_cols

    # ---------- 유틸: 배열→DF ----------
    def _to_df(X: Optional[np.ndarray], F: Optional[np.ndarray], G: Optional[np.ndarray]) -> Optional[pd.DataFrame]:
        if X is None or F is None:
            return None
        x_cols, f_cols, g_cols = _names(problem)
        dfs = [pd.DataFrame(X, columns=x_cols),
               pd.DataFrame(F, columns=f_cols)]
        if G is not None:
            if G.ndim == 1:
                G = G.reshape(-1, 1)
            # g열 수 불일치시 generic 이름으로 대체
            cols = g_cols if len(g_cols) == G.shape[1] else [f"g::{i+1}" for i in range(G.shape[1])]
            dfs.append(pd.DataFrame(G, columns=cols))
        return pd.concat(dfs, axis=1)

    # ---------- 1) feasible Pareto에서 1해 선택 ----------
    opt = getattr(res, "opt", None)
    if opt is not None and len(opt) > 0:
        Xo, Fo, Go = opt.get("X"), opt.get("F"), opt.get("G")
        df = _to_df(Xo, Fo, Go)
        if df is not None and not df.empty:
            f_cols = [c for c in df.columns if c.startswith("f::")]
            # ∑F 최소 1개 선택
            best_idx = int(df[f_cols].sum(axis=1).values.argmin())
            out = df.iloc[[best_idx]].copy()
            out["note"] = "feasible"
            return out

    # ---------- 2) fallback: least-infeasible 1해 선택 ----------
    pop = getattr(getattr(res, "algorithm", None), "pop", None)
    if pop is None or len(pop) == 0:
        return pd.DataFrame()

    Xp, Fp, Gp = pop.get("X"), pop.get("F"), pop.get("G")
    if Xp is None or Fp is None or len(Fp) == 0:
        return pd.DataFrame()

    Xp = np.asarray(Xp, float)
    Fp = np.asarray(Fp, float)

    # CV = ∑ max(G,0); G 없으면 0
    if Gp is not None:
        Gp = np.asarray(Gp, float)
        CV = np.maximum(Gp, 0.0).sum(axis=1)
    else:
        CV = np.zeros(len(Fp), float)

    # 합성 목적 Y = [CV, F1, F2, ...] (모두 최소화)
    Y = np.column_stack([CV, Fp])

    # NDS로 front-0 인덱스
    nds = NonDominatedSorting()
    fronts = nds.do(Y, only_non_dominated_front=False)
    front0 = np.array(fronts[0], dtype=int)
    if front0.size == 0:
        return pd.DataFrame()

    # front-0 안에서 CV → ∑F 오름차순으로 1개 선택
    fsum0 = Fp[front0].sum(axis=1)
    order0 = np.lexsort((fsum0, CV[front0]))  # 1st: CV, 2nd: ∑F
    best = front0[order0[0]]

    # 최종 1행 DF 생성
    df_fb = _to_df(Xp[[best]], Fp[[best]], (Gp[[best]] if Gp is not None else None))
    if df_fb is None:
        return pd.DataFrame()
    df_fb["note"] = "least_infeasible"
    return df_fb

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from loguru import logger


def _pick_decision_from_pareto(pareto_df: pd.DataFrame, important_features: List[str]) -> Dict[str, float]:
    """pareto_df(1행)에서 x::feat → 의사결정변수 dict로 변환."""
    row = pareto_df.iloc[0]
    dec: Dict[str, float] = {}
    for f in important_features:
        col = f"x::{f}"
        if col in row.index:
            dec[f] = float(row[col])
    return dec


def _apply_decision(for_optimal: pd.DataFrame, decision: Dict[str, float], feature_names: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """결정변수 적용 전/후 입력 X를 반환."""
    X_before = for_optimal[feature_names].copy()
    X_after = X_before.copy()
    for k, v in decision.items():
        if k in X_after.columns:
            X_after[k] = v  # 전 행 동일 적용
    return X_before, X_after


def _predict_all(models: Dict[str, Any], X: pd.DataFrame, feature_names: List[str]) -> Dict[str, np.ndarray]:
    """타깃별 예측."""
    preds: Dict[str, np.ndarray] = {}
    for t, model in models.items():
        y = model.predict(X[feature_names])
        preds[t] = np.asarray(y, dtype=float).ravel()
    return preds


def export_mo_report_csv(
    *,
    out_mo: Dict[str, Any],
    for_optimal: pd.DataFrame,
    models: Dict[str, Any],
    feature_names: List[str],
    important_features: List[str],
    spec_map_keys: List[str],                  # 예: ["Discharge","Pressure",...]
    y_true_map: Dict[str, pd.Series],          # 타깃별 y_test (for_optimal 인덱스와 일치)
    save_path: Path,
) -> Path:
    """MO 최종 해를 적용하여 타깃별 전/후 예측과 실측을 CSV로 저장.

    결과 컬럼:
      - idx, target
      - y_true, y_pred_before, y_pred_after, delta_after(=after - y_true)
      - mae_before, mae_after
      - (선택) 결정변수 스냅샷: x::<feat> 컬럼들(맨 앞 1행만; 참고용)
    """
    logger.info("Preparing multi-objective report CSV...")

    # 1) 최종 해(한 행) 가져오기
    mo_res = out_mo["results"]["mo_cat"]  # 필요시 키 변경
    pareto_df: pd.DataFrame = mo_res.pareto_df
    assert len(pareto_df) == 1, "pareto_df는 최종 1행이어야 합니다."

    decision = _pick_decision_from_pareto(pareto_df, important_features)
    logger.info(f"Chosen decision (|V|={len(decision)}): {decision}")

    # 2) 전/후 입력 구성
    X_before, X_after = _apply_decision(for_optimal, decision, feature_names)

    # 3) 타깃별 예측
    pred_before = _predict_all(models, X_before, feature_names)
    pred_after  = _predict_all(models,  X_after,  feature_names)

    # 4) 타깃별 행결합 리포트
    frames: List[pd.DataFrame] = []
    for t in spec_map_keys:
        if t not in models:
            continue
        y_true = y_true_map[t].reindex(for_optimal.index).astype(float)
        y_bef  = pred_before[t]
        y_aft  = pred_after[t]

        df_t = pd.DataFrame({
            "idx": for_optimal.index,
            "target": t,
            "y_true": y_true.to_numpy(),
            "y_pred_before": y_bef,
            "y_pred_after":  y_aft,
        })
        # 오차 요약(행 단위)
        df_t["delta_after"] = df_t["y_pred_after"] - df_t["y_true"]
        df_t["mae_before"]  = np.abs(df_t["y_pred_before"] - df_t["y_true"])
        df_t["mae_after"]   = np.abs(df_t["y_pred_after"]  - df_t["y_true"])
        frames.append(df_t)

    report = pd.concat(frames, axis=0, ignore_index=True)

    # 5) 결정변수 스냅샷(참고): x::feat 컬럼을 맨 앞 블록으로 별도 섹션처럼 덧붙임
    #    - 데이터 분석 편의를 위해 1행만 기록(선택)
    x_cols = [f"x::{f}" for f in important_features]
    x_row = {f"x::{k}": v for k, v in decision.items()}
    x_block = pd.DataFrame([x_row], columns=x_cols)
    sep = pd.DataFrame([{"idx": "", "target": "", "y_true": "", "y_pred_before": "", "y_pred_after": "",
                         "delta_after": "", "mae_before": "", "mae_after": ""}])

    # 6) 저장 (결정변수 스냅샷 → 빈 줄 → 본문)
    save_path = Path(save_path).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        x_block.to_csv(f, index=False)
        sep.to_csv(f, index=False, header=False)
        report.to_csv(f, index=False)

    logger.info(f"Saved MO report → {save_path}")
    return save_path
