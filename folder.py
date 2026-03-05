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


import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, Union, List

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

# --- [여기서부터 1:N 및 조건 필터링 안전 업데이트] ---

# 1. P960_PARAM이 포함된 파일 경로 안전하게 제외 (단순 문자열 매칭)
valid_mask = ~full_path_in.str.contains("P960_PARAM", na=False, regex=False)

# 필터링이 적용된 리스트 추출
filtered_job_codes = job_code_in[valid_mask].tolist()
filtered_full_paths = full_path_in[valid_mask].tolist()

# 2. 중복을 허용하여 리스트로 안전하게 모아주는 임시 딕셔너리 생성
raw_job_dict = defaultdict(list)

# 기존에 쓰시던 zip 방식 그대로 순회하면서 리스트에 경로를 차곡차곡 누적
for j_code, f_path in zip(filtered_job_codes, filtered_full_paths):
    # 만약 완전히 동일한 경로가 중복해서 들어오는 것을 방지
    if f_path not in raw_job_dict[j_code]:
        raw_job_dict[j_code].append(f_path)

# 3. 최종 규격(단일: str, 복수: list[str])에 맞추어 변환
job_folder_dict: Dict[str, Union[str, List[str]]] = {}

for j_code, paths in raw_job_dict.items():
    if len(paths) == 1:
        job_folder_dict[j_code] = paths[0]  # 파일이 1개면 순수 문자열로 할당
    else:
        job_folder_dict[j_code] = paths     # 2개 이상이면 리스트 통째로 할당
