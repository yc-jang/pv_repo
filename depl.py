from pathlib import Path
from typing import Dict, Iterable, Tuple, List


def build_keyword_path_map(
    models_dir: Path | str,
    keywords: Iterable[str],
    suffixes: Tuple[str, ...] = (".pkl",),
    match_mode: str = "auto",
    case_sensitive: bool = False,
) -> Dict[str, Path]:
    """models/ 하위 파일들을 KEYWORD와 1:1 매칭해 경로 매핑을 만든다.

    파일명(stem)과 KEYWORD의 일치 규칙:
      - match_mode="exact": 파일의 stem이 KEYWORD와 정확히(대소문자 옵션) 일치
      - match_mode="prefix": 파일의 stem이 KEYWORD로 시작
      - match_mode="auto"  : 우선 exact 시도 후 실패 시 prefix로 단일 매칭이면 채택

    Args:
        models_dir: 모델 파일들이 위치한 디렉토리(예: "models").
        keywords: 필요한 타깃명 목록. 각 항목이 정확히 하나의 파일과 매칭되어야 한다.
        suffixes: 허용 파일 확장자(tuple). 기본은 (".pkl",).
        match_mode: "exact" | "prefix" | "auto".
        case_sensitive: 대소문자 구분 여부. 기본 False(비구분).

    Returns:
        {keyword: 상대경로(Path)} 딕셔너리. 상대경로는 현재 작업 디렉토리 기준
        (예: models/BET.pkl).

    Raises:
        FileNotFoundError: models_dir가 없거나 비어 있을 때.
        ValueError: 다음 상황
            - KEYWORD 중 일부가 매칭되지 않음
            - 어떤 KEYWORD에 대해 후보가 2개 이상(모호)
            - 하나의 파일이 둘 이상의 KEYWORD에 매칭(충돌)
    """
    # 핵심 원리: 디렉토리 내 허용 확장자 파일 수집 → 키워드별 후보 선택 → 1:1 검증
    mdir = Path(models_dir)
    if not mdir.exists() or not mdir.is_dir():
        raise FileNotFoundError(f"Models directory not found: {mdir}")

    files: List[Path] = [
        p for p in sorted(mdir.iterdir())
        if p.is_file() and p.suffix.lower() in {s.lower() for s in suffixes}
    ]
    if not files:
        raise FileNotFoundError(f"No model files with {suffixes} found under: {mdir}")

    # 파일 stem 전처리(대소문 옵션)
    def norm(s: str) -> str:
        return s if case_sensitive else s.lower()

    stem_to_paths: Dict[str, List[Path]] = {}
    for p in files:
        stem = norm(p.stem)
        stem_to_paths.setdefault(stem, []).append(p)

    # 중복 stem(동일 stem, 다른 확장자 등) 방지 체크
    for stem, paths in stem_to_paths.items():
        if len(paths) > 1:
            # 필요 시 정책 변경 가능: 여기선 애초에 모호하므로 에러
            raise ValueError(f"Ambiguous files for stem '{stem}': {[pp.name for pp in paths]}")

    used_paths: set[Path] = set()
    result: Dict[str, Path] = {}

    for kw in keywords:
        kw_n = norm(kw)

        # 1) exact 매칭
        exact_path = stem_to_paths.get(kw_n, [None])[0] if kw_n in stem_to_paths else None

        candidates: List[Path] = []
        if exact_path is not None:
            candidates = [exact_path]
        else:
            # 2) prefix 매칭(요청/설정에 따라)
            if match_mode in ("prefix", "auto"):
                for stem, path_list in stem_to_paths.items():
                    if stem.startswith(kw_n):
                        candidates.extend(path_list)

        # 후보 평가
        if match_mode == "exact" and exact_path is None:
            raise ValueError(f"No exact match for keyword '{kw}' in {mdir}")

        if not candidates:
            raise ValueError(f"No file matched for keyword '{kw}' in {mdir}")

        if len(candidates) > 1:
            # 모호: 동일 키워드에 여러 파일 충돌
            names = [c.name for c in candidates]
            raise ValueError(f"Ambiguous matches for keyword '{kw}': {names}")

        chosen = candidates[0]

        # 하나의 파일이 두 KEYWORD에 중복 매칭되는지 체크
        if chosen in used_paths:
            raise ValueError(
                f"File '{chosen.name}' matched by multiple keywords, "
                f"including '{kw}'. Ensure 1:1 mapping."
            )

        used_paths.add(chosen)
        # 상대경로(현재 작업 디렉토리 기준)로 저장
        result[kw] = (Path(mdir.name) / chosen.name)

    # 모든 KEYWORD가 1:1로 매핑되었는지 최종 검증(파일 수 == 키워드 수)
    if len(result) != len(list(keywords)):
        raise ValueError("Keyword-to-file mapping incomplete or inconsistent.")

    return result
