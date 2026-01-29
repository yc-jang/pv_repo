import pandas as pd
from pathlib import Path

# --- (A) 저장폴더 갱신 (SAGGAR/PIPE는 유지) ---
# 전제: base는 이미 "/AI_T_DATA/..." 형태로 준비되어 있다고 가정 (C: 제거 등 완료)
src = param_csv["저장폴더"].astype("string").str.replace("\\", "/", regex=False)

# '/IN/' 기준으로만 안전하게 상대경로 추출
rel = src.str.split("/IN/", n=1).str[-1].astype("string").str.lstrip("/")

fname = param_csv["파일명"].astype("string")
mask_keep = fname.str.contains(r"(SAGGAR|PIPE)", case=True, regex=True, na=False)

param_csv.loc[~mask_keep, "저장폴더"] = (base.rstrip("/") + "/" + rel).astype(str)

# --- (B) job_folder_dict 생성 ---
# 대상: IN 행만
in_mask = param_csv["입출력구분"].astype("string").eq("IN")

# 기본룰: rel의 첫 2개 폴더의 "선행 숫자 1자리"를 결합 -> 예: 0HEAT/1LOTC => "01"
rel_norm = rel.astype("string").str.replace("\\", "/", regex=False).str.lstrip("/")
seg0 = rel_norm.str.split("/", n=2).str[0]
seg1 = rel_norm.str.split("/", n=2).str[1]

d0 = seg0.str.extract(r"^(\d)")[0]
d1 = seg1.str.extract(r"^(\d)")[0]
base_code = (d0.fillna("") + d1.fillna("")).astype("string")

# IN인데 기본룰 코드가 안 만들어지는 경우는 데이터/규칙 불일치 -> 즉시 에러
bad_base = in_mask & (base_code.eq("") | d0.isna() | d1.isna())
if bad_base.any():
    sample = param_csv.loc[bad_base, ["저장폴더", "파일명", "입출력구분"]].head(10)
    raise ValueError(f"기본룰 코드 생성 실패(IN 행). 예시:\n{sample}")

job_code = base_code

# 예외룰 우선순위: LOT_TRACKING_2(211) > SAGGAR(91) > PIPE(92) > 기본룰
job_code = job_code.mask(fname.str.contains("PIPE", na=False), "92")
job_code = job_code.mask(fname.str.contains("SAGGAR", na=False), "91")
job_code = job_code.mask(fname.str.contains("LOT_TRACKING_2", na=False), "211")

job_code_in = job_code[in_mask].astype(str)

# value: (갱신된) 저장폴더 + "/" + 파일명 (전체 경로)
folder_norm = param_csv["저장폴더"].astype("string").str.replace("\\", "/", regex=False).str.rstrip("/")
file_norm = fname.str.replace("\\", "/", regex=False).str.lstrip("/")
full_path_in = (folder_norm + "/" + file_norm)[in_mask].astype(str)

# 코드 중복이면 ValueError
vc = job_code_in.value_counts()
dup_codes = vc[vc > 1].index.tolist()
if dup_codes:
    raise ValueError(f"job code 중복 발생: {dup_codes}")

job_folder_dict: dict[str, str] = dict(zip(job_code_in.tolist(), full_path_in.tolist()))
