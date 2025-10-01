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
