"""[공통] 폴더 규칙·경로 헬퍼 — 이 폴더 안 모든 단계가 여기 정의만 본다.

폴더 구조 (이 파일이 있는 곳 = ROOT):
  music/                       ← 원곡(풀 믹스) wav 를 넣는 곳 (music/raw/ 도 인식)
  분석/<곡>/                    allin1 결과(json) + demucs 4-stem + <곡>_vocals.wav + <곡>_MR.wav
  궤적/<곡>/
     1_오프라인궤적/             head + mouth            (조명 X, 비브라토 X)
     2_오프라인궤적+조명/        head + mouth + led      (비브라토 X)
     3_전체_비브라토포함/         head + mouth(비브라토) + led   ← Pi 배포 기본
     _중간산출물/                원본 산출물·디버그 csv·요약 txt
  data/segments_offline/       헤드 세그먼트 라이브러리(.npy) — 헤드 궤적 생성 필수
  data/pyin_cache/             pyin f0 캐시(비브라토 검출 가속, 자동 생성)
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
MUSIC_DIR = ROOT / "music"
ANALYSIS_DIR = ROOT / "분석"
TRAJ_DIR = ROOT / "궤적"
DATA_DIR = ROOT / "data"

# 단계 폴더 이름 (누적: 1 ⊂ 2 ⊂ 3)
STAGES = {
    1: "1_오프라인궤적",
    2: "2_오프라인궤적+조명",
    3: "3_전체_비브라토포함",
}
INTERMEDIATE_DIRNAME = "_중간산출물"

# Pi 재생 계약 파일명 (<곡> 부분이 4채널 모두 동일해야 한 곡으로 묶임)
def head_csv_name(song: str) -> str:
    return f"{song}.csv"


def mouth_csv_name(song: str) -> str:
    return f"{song}-delta-big.csv"


def led_csv_name(song: str) -> str:
    return f"{song}-led.csv"


def list_raw_songs() -> list[Path]:
    """music/ (및 music/raw/) 안의 원곡 wav 목록. 파일명 정렬."""
    found: dict[str, Path] = {}
    for d in (MUSIC_DIR, MUSIC_DIR / "raw"):
        if d.is_dir():
            for p in sorted(d.glob("*.wav")):
                found.setdefault(p.stem, p)
    return [found[k] for k in sorted(found)]


def resolve_raw(arg: str) -> Path:
    """곡 이름(확장자 유무 무관) 또는 wav 경로 → 실제 파일 경로."""
    p = Path(arg)
    if p.exists() and p.is_file():
        return p
    name = Path(arg).stem
    for d in (MUSIC_DIR, MUSIC_DIR / "raw"):
        for cand in (d / f"{name}.wav", d / arg):
            if cand.exists() and cand.is_file():
                return cand
    raise FileNotFoundError(f"원곡을 찾을 수 없음: {arg}  (music/ 에 <곡>.wav 를 넣으세요)")


def analysis_dir(song: str) -> Path:
    return ANALYSIS_DIR / song


def traj_dir(song: str) -> Path:
    return TRAJ_DIR / song


def stage_dir(song: str, k: int) -> Path:
    return traj_dir(song) / STAGES[k]


def intermediate_dir(song: str) -> Path:
    return traj_dir(song) / INTERMEDIATE_DIRNAME


def win_to_wsl(p: Path | str) -> str:
    """C:\\Users\\... → /mnt/c/Users/...  (WSL 경로)."""
    p = Path(p).resolve()
    drive = p.drive.rstrip(":").lower()
    rest = str(p)[len(p.drive):].replace("\\", "/")
    return f"/mnt/{drive}{rest}"


def utf8_stdout():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
