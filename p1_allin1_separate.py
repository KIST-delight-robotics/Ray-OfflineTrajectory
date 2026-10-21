"""[1단계] allin1 구조 분석 + 음원 분리 (WSL 안에서 실행, Windows 에서 호출).

원곡 wav 하나 → 분석/<곡>/ 아래에:
  <곡>.json                 allin1 결과: bpm / beats / downbeats / segments(intro·verse·chorus …)
  stems/{vocals,drums,bass,other}.wav   allin1 이 내부에서 돌린 demucs(htdemucs 4-stem) 부산물
  <곡>_vocals.wav           = stems/vocals.wav 복사       → 입/헤드/비브라토 궤적 입력
  <곡>_MR.wav               = drums+bass+other 합         → (참고용 반주; LED 는 stem 을 직접 씀)
  <곡>_allin1_summary.txt   사람이 읽는 요약(BPM·구간표)

실행 환경:
  natten(allin1 의존)이 Windows 휠이 없어 WSL Ubuntu-22.04 의 venv(/root/allin1env, GPU) 에서 돈다. 환경 생성: wsl_setup/setup_allin1env.bat
  GPU 기준 곡당 약 1분(3분 곡, RTX 3050). 결과는 분석/<곡>/ 에 남으므로 2회차부터는 즉시.

사용:
  python p1_allin1_separate.py Butter_BTS            # music/Butter_BTS.wav 분석+분리
  python p1_allin1_separate.py Butter_BTS --refresh  # 캐시 무시하고 다시
파이프라인에서는 analyze_and_separate(song, raw_wav) 호출.
"""
from __future__ import annotations

import argparse
import json
import shutil
import os
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from common import analysis_dir, resolve_raw, utf8_stdout, win_to_wsl

WSL_DISTRO = os.environ.get("VIBRATO_WSL_DISTRO", "Ubuntu-22.04")  # 배포판 이름이 다르면 환경변수로
WSL_VENV = "/root/allin1env"
WSL_PY = f"{WSL_VENV}/bin/python"
WSL_ALLIN1 = f"{WSL_VENV}/bin/allin1"

STEM_NAMES = ("vocals", "drums", "bass", "other")

# 사내망 SSL 인터셉트 프록시 대응(모델 가중치 첫 다운로드 시 필요). 이미 등록된 신뢰소를 쓰게 고정.
_CERT = "/etc/ssl/certs/ca-certificates.crt"
_ENVP = (f"export PIP_CERT={_CERT} SSL_CERT_FILE={_CERT} "
         f"REQUESTS_CA_BUNDLE={_CERT} CURL_CA_BUNDLE={_CERT}; ")


def _wsl(cmd: str, timeout: int | None = None):
    """WSL 안에서 bash 한 줄 실행 (멀티라인은 CRLF 로 깨지므로 && 체인으로만)."""
    full = ["wsl", "-d", WSL_DISTRO, "-u", "root", "bash", "-lc", _ENVP + cmd]
    return subprocess.run(full, capture_output=True, text=True, timeout=timeout,
                          encoding="utf-8", errors="replace")


def wsl_env_ready() -> bool:
    try:
        r = _wsl(f"test -x {WSL_ALLIN1} && {WSL_PY} -c 'import allin1' && echo OK", timeout=180)
        return "OK" in (r.stdout or "")
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────
#  결과 파일 규칙
# ──────────────────────────────────────────────────────────────────────
def _paths(song: str) -> dict:
    d = analysis_dir(song)
    return {
        "dir": d,
        "json": d / f"{song}.json",
        "stems": d / "stems",
        "vocals": d / f"{song}_vocals.wav",
        "mr": d / f"{song}_MR.wav",
        "summary": d / f"{song}_allin1_summary.txt",
        "demix_tmp": d / "_demix",
    }


def _stems_complete(stem_dir: Path) -> bool:
    return all((stem_dir / f"{n}.wav").exists() for n in STEM_NAMES)


def _load_json(json_path: Path) -> dict:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return {
        "bpm": float(data.get("bpm", 0.0)),
        "beats": [float(b) for b in data.get("beats", [])],
        "downbeats": [float(b) for b in data.get("downbeats", [])],
        "segments": [{"start": float(s["start"]), "end": float(s["end"]), "label": str(s["label"])}
                     for s in data.get("segments", [])],
    }


# ──────────────────────────────────────────────────────────────────────
#  allin1 실행
# ──────────────────────────────────────────────────────────────────────
def _run_allin1(song: str, raw_wav: Path, P: dict, overwrite: bool) -> None:
    if not wsl_env_ready():
        raise RuntimeError(
            f"WSL allin1 환경이 없음 ({WSL_DISTRO}:{WSL_VENV}). "
            "wsl_setup/setup_allin1env.bat 로 환경을 만드세요 (README 참고).")
    P["dir"].mkdir(parents=True, exist_ok=True)
    wsl_wav = win_to_wsl(raw_wav)
    wsl_out = win_to_wsl(P["dir"])
    wsl_demix = win_to_wsl(P["demix_tmp"])
    ow = " --overwrite" if overwrite else ""
    print(f"  allin1 분석 중 (WSL/GPU): {song} ...")
    r = _wsl(f"cd /tmp && {WSL_ALLIN1} -o '{wsl_out}' --demix-dir '{wsl_demix}' "
             f"--spec-dir /tmp/allin1_spec --keep-byproducts{ow} '{wsl_wav}'", timeout=3 * 3600)
    if r.returncode != 0 or not P["json"].exists():
        raise RuntimeError(f"allin1 실패: {(r.stderr or r.stdout or '')[-800:]}")
    # demucs stem 을 stems/ 로 정리 (htdemucs/<곡>/*.wav)
    src = P["demix_tmp"] / "htdemucs" / raw_wav.stem
    if not src.exists():
        raise RuntimeError(f"demucs stem 출력 누락: {src}")
    P["stems"].mkdir(parents=True, exist_ok=True)
    for st in src.glob("*.wav"):
        st.replace(P["stems"] / st.name)
    shutil.rmtree(P["demix_tmp"], ignore_errors=True)


# ──────────────────────────────────────────────────────────────────────
#  vocals / MR 배치
# ──────────────────────────────────────────────────────────────────────
def _make_vocals_and_mr(P: dict) -> None:
    stems = P["stems"]
    if not P["vocals"].exists():
        shutil.copy2(stems / "vocals.wav", P["vocals"])   # PCM_16 그대로(입 궤적 코드가 int wav 기대)
    if not P["mr"].exists():
        acc = None
        sr = None
        for n in ("drums", "bass", "other"):
            y, sr_i = sf.read(str(stems / f"{n}.wav"), dtype="float32", always_2d=True)
            if acc is None:
                acc, sr = y, sr_i
            else:
                m = min(len(acc), len(y))
                acc = acc[:m] + y[:m]
        acc = np.clip(acc, -1.0, 1.0)
        sf.write(str(P["mr"]), acc, sr, subtype="PCM_16")


def _write_summary(song: str, A: dict, P: dict, source: str) -> None:
    ko = {"start": "시작", "intro": "인트로", "verse": "벌스", "chorus": "후렴", "bridge": "브릿지",
          "inst": "간주", "solo": "솔로", "outro": "아웃트로", "break": "브레이크", "end": "끝"}
    lines = [f"♪ {song}  — allin1 구조 분석 요약  (출처: {source})",
             f"BPM {A['bpm']:.1f} | beats {len(A['beats'])}개 | downbeats {len(A['downbeats'])}개",
             "", "구간(start~end s, 라벨):"]
    for s in A["segments"]:
        lines.append(f"  {s['start']:6.1f} ~ {s['end']:6.1f}  {ko.get(s['label'], s['label'])}({s['label']})")
    lines += ["", "파일:", f"  vocals: {P['vocals'].name}", f"  MR    : {P['mr'].name}",
              f"  stems : {', '.join(n + '.wav' for n in STEM_NAMES)}"]
    P["summary"].write_text("\n".join(lines), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────
#  공용 진입점
# ──────────────────────────────────────────────────────────────────────
def analyze_and_separate(song: str, raw_wav: Path, *, refresh: bool = False) -> dict:
    """곡 분석+분리 (캐시 우선). 반환:
        bpm, beats, downbeats, segments[{start,end,label}],
        stems{name: Path}, vocals(Path), mr(Path), json(Path), source(str)
    """
    P = _paths(song)
    source = "cache"
    if refresh or not (P["json"].exists() and _stems_complete(P["stems"])):
        if refresh:
            for k in ("json", "vocals", "mr", "summary"):
                if P[k].exists():
                    P[k].unlink()
            shutil.rmtree(P["stems"], ignore_errors=True)
        _run_allin1(song, raw_wav, P, overwrite=refresh)
        source = "allin1"
    else:
        print(f"  [캐시] 분석/{song}/ 이미 있음 → 재사용 (다시 하려면 --refresh)")

    _make_vocals_and_mr(P)
    A = _load_json(P["json"])
    A["stems"] = {n: P["stems"] / f"{n}.wav" for n in STEM_NAMES}
    A["vocals"] = P["vocals"]
    A["mr"] = P["mr"]
    A["json"] = P["json"]
    A["source"] = source
    _write_summary(song, A, P, source)
    return A


def main():
    utf8_stdout()
    ap = argparse.ArgumentParser(description="[1단계] allin1 분석 + demucs 분리")
    ap.add_argument("song", help="곡 이름 또는 wav 경로")
    ap.add_argument("--refresh", action="store_true", help="캐시 무시하고 다시 분석")
    args = ap.parse_args()
    raw = resolve_raw(args.song)
    analyze_and_separate(raw.stem, raw, refresh=args.refresh)
    print(_paths(raw.stem)["summary"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
