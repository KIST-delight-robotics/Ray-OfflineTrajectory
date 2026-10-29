"""[3단계] back 조명(LED 막대) 밝기 궤적 — allin1 구간 라벨 기반. LED_MODE 로 연출 선택.

[final_chorus] (기본) 마지막 후렴에서만 점등, 나머지 구간은 0.
  - 마지막 후렴 = 마지막 chorus 와 바로 앞에 이어진 chorus 구간들을 합친 덩어리 (없으면 전체 소등)
  - 진입 → 4s 밝아짐(→1.0) → 1.5s 피크 유지 → 3s 내려옴 → 유지 곡선 → 후렴 끝 후 4s 꺼짐(곡 끝엔 반드시 0)
  - 유지 곡선 = 중심 0.38 ± 0.08, 곡 음량(4-stem 합 RMS, 0.5s 평활)에 따라 완만히 출렁 — 박자 깜박임 없음
  - 성분 분해 b = PEAK·p + w·(C + A·m) → <곡>-led_parts.csv(p,w,m) 로 함께 내보내 재생 쪽에서 C·A·PEAK·감마 재조합

[allin1] (이전 기본, 태그 v1-led-allin1) 설계 철학: "곡의 구조(설계도)가 곧 연출이다."
  - 둥둥도(mix): allin1 구간 라벨이 직접 지정 (후렴=폭발, 벌스=절제, 끝=페이드)
  - 베이스 높낮이: 반주 stem(drums+bass+other) RMS 와 라벨 레벨의 절충
  - 둥둥 펄스: allin1 beats/downbeats 에 락, 세기 = drums stem 온셋
  - 드랍(둥둥도 큰 점프) 1마디 전 암전 → 버스트 (조명 오퍼레이터 관례)
  - 후렴 회차 에스컬레이션(1절 후렴 82% → 마지막 100%)
  - 광과민성 안전: 펄스 최소 간격 150ms(3Hz), 저장 시 지각 감마 1.8

출력 계약(Pi): <곡>-led.csv — 1열 밝기(0~1), 헤더 없음, 40ms/행(25fps), 입/헤드와 같은 행 수.
         <곡>-led_parts.csv — final_chorus 모드만. 헤더 p,w,m + 같은 행 수, 감마 미적용.
(모터 ID6 은 하드웨어에서 제거됨 → 막대 조향 없음, 밝기만.)

파이프라인에서는 generate_led(song, analysis, raw_wav, n_frames) → write_led_csv(...) [+ write_led_parts_csv(...)] 호출.
단독: python p3_led_allin1.py Butter_BTS   (분석/<곡>/ 이 있어야 함)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from common import intermediate_dir, led_csv_name, resolve_raw, utf8_stdout

EPS = 1e-8

# ════════════════════════════════════════════════════════════════════
#  튜닝 손잡이
# ════════════════════════════════════════════════════════════════════
# 연출 모드: "final_chorus" = 마지막 후렴에서만 점등(유지 구간만 음량 따라 완만히 출렁)
#            "allin1"       = 기존 설계도 연출(라벨·박자 둥둥)
LED_MODE = "final_chorus"

# ── final_chorus 모드 ──
# 마지막 후렴(연속 chorus 구간 병합) 진입 → 밝아짐 → 최대 잠깐 유지 → 내려옴 → 유지 → 후렴 끝나고 꺼짐
FC_PEAK = 1.0                 # 최대 밝기
FC_HOLD_LEVEL = 0.38          # 후렴 유지 밝기(설계값; CSV 는 감마 적용)
FC_RISE_S = 4.0               # 0 → 최대
FC_PEAK_HOLD_S = 1.5          # 최대 유지
FC_FALL_S = 3.0               # 최대 → 유지 레벨
FC_FADE_OUT_S = 4.0           # 후렴 끝 → 0 (곡 끝까지 여유가 모자라면 그만큼 줄임)
FC_FADE_OUT_MIN_S = 1.5       # 여유가 이보다 짧으면 후렴 끝 전부터 미리 꺼짐 시작
FC_INTRO_MAX_FRAC = 0.5       # 밝아짐~내려옴이 후렴 길이의 이 비율을 넘으면 비례 축소
# 유지 구간 움직임: 유지 레벨을 중심으로, 곡 음량(전체 stem 합 RMS)에 따라 위아래로 출렁
FC_MOVE = True
FC_MOVE_AMP = 0.08            # 중심(유지 레벨) ± 진폭(설계값) → 0.30~0.46
FC_MOVE_SMOOTH_S = 0.5        # 음량 평활(초) — 박마다 깜박이지 않고 프레이즈 단위로 숨쉬기
FC_MOVE_PCT = (10.0, 90.0)    # 마지막 후렴 안에서 음량 정규화 분위수(하, 상) — 이 폭이 ±진폭

# ── allin1 모드 ──
# 구간 라벨 → (둥둥도 mix 0~1, 베이스 레벨 0~1) — 설계도가 직접 연출 지정
LABEL_PLAN = {
    "chorus": (0.85, 0.95),
    "verse":  (0.40, 0.55),
    "inst":   (0.55, 0.65),
    "solo":   (0.60, 0.70),
    "bridge": (0.35, 0.50),
    "intro":  (0.20, 0.35),
    "outro":  (0.10, 0.25),
    "break":  (0.30, 0.40),
    "start":  (0.10, 0.20),
    "end":    (0.05, 0.15),
}
PLAN_DEFAULT = (0.40, 0.55)   # 모르는 라벨 → 벌스 취급
PLAN_RAMP_S = 1.2             # 구간 전환 부드럽게(초)
CHORUS_ESC_START = 0.82       # 후렴 회차 에스컬레이션 시작 비율(첫 후렴), 마지막=1.0

# 베이스(파도)
BASE_W_LABEL = 0.5            # 라벨 레벨 비중(나머지는 stem 에너지)
DIM_FLOOR, DIM_CEIL = 0.0, 0.70
WAVE_AMP = 0.32               # 파도 출렁 진폭(mix 낮을수록 도드라짐)
WAVE_BARS1, WAVE_BARS2 = 4.0, 6.0   # 출렁 주기(마디 수)
ENERGY_SMOOTH_S = 3.0         # stem 에너지 평활(초)

# 둥둥 펄스 — 박자 락
DOWNBEAT_GAIN = 1.0
BEAT_GAIN = 0.72
DRUM_STR_PCT = 90.0           # drums 온셋세기 정규화 분위수
FLASH_ATTACK_S = 0.05
FLASH_RELEASE_BEATS = 0.5     # 꼬리 = 박자 비례
FLASH_RELEASE_MIN, FLASH_RELEASE_MAX = 0.10, 0.32
MIN_PULSE_INTERVAL_S = 0.15   # 초당 3회 이상 점멸 회피

# 블렌드
DIM_DUCK = 0.65               # 둥둥 강한 구간 베이스 골 낮춤
BRIGHTNESS_GAIN = 1.20        # 전체 밝기 게인

# 드랍 선행 연출(암전→버스트)
DROP_JUMP_MIN = 0.25          # "드랍"으로 인정할 둥둥도 점프량
ANTICIPATION_DEPTH = 0.85     # 드랍 직전 얼마나 가라앉힐지(0.85=15%까지)
ANTICIPATION_BARS = 1.0       # 디밍 시작 = 드랍 몇 마디 전

# 저장 시 지각 감마(하드웨어 듀티 보정; 설계값은 그대로, CSV 만 보정)
LED_OUT_GAMMA = 1.8

_KO = {"start": "시작", "intro": "인트로", "verse": "벌스", "chorus": "후렴",
       "bridge": "브릿지", "inst": "간주", "solo": "솔로", "outro": "아웃트로",
       "break": "브레이크", "end": "끝"}


# ════════════════════════════════════════════════════════════════════
#  수치 헬퍼
# ════════════════════════════════════════════════════════════════════
def _gaussian_smooth(values: np.ndarray, times_s: np.ndarray, smoothing_s: float) -> np.ndarray:
    if values.size <= 2 or smoothing_s <= 0.0 or times_s.size <= 1:
        return values
    dt = float(np.median(np.diff(times_s)))
    if dt <= 0.0:
        return values
    radius = max(1, int(round(smoothing_s / dt)))
    if radius <= 1:
        return values
    x = np.arange(-radius, radius + 1, dtype=float)
    sigma = max(radius / 2.5, EPS)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= np.sum(kernel)
    return np.convolve(values, kernel, mode="same")


def _resample_to_grid(values: np.ndarray, src_t: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros_like(target_t, dtype=float)
    return np.interp(target_t, src_t, values, left=values[0], right=values[-1])


def _min_interval_filter(times: np.ndarray, strengths: np.ndarray, min_s: float = MIN_PULSE_INTERVAL_S):
    """너무 촘촘한 펄스 솎아내기 — min_s 안에 겹치면 센 놈만."""
    if times.size == 0:
        return times, strengths
    keep_t, keep_s = [float(times[0])], [float(strengths[0])]
    for t, s in zip(times[1:], strengths[1:]):
        if t - keep_t[-1] < min_s:
            if s > keep_s[-1]:
                keep_t[-1], keep_s[-1] = float(t), float(s)
        else:
            keep_t.append(float(t))
            keep_s.append(float(s))
    return np.array(keep_t), np.array(keep_s)


def apply_drop_anticipation(brightness: np.ndarray, drop_times, bpm: float, dt: float,
                            depth: float = ANTICIPATION_DEPTH) -> np.ndarray:
    """드랍마다: 1마디 전부터 선형 디밍 → 직전 반 박 최저 유지 → 드랍에서 즉시 해제(버스트)."""
    n = brightness.size
    env = np.ones(n)
    bar_s = 4.0 * 60.0 / max(bpm, 1.0)
    beat_s = 60.0 / max(bpm, 1.0)
    for t_d in drop_times:
        t0 = t_d - ANTICIPATION_BARS * bar_s
        t1 = t_d - 0.5 * beat_s
        i0, i1, id_ = int(t0 / dt), int(t1 / dt), int(t_d / dt)
        i0, i1, id_ = max(0, i0), max(0, i1), max(0, min(n, id_))
        if id_ <= i0 or i1 <= i0:
            continue
        ramp = np.linspace(1.0, 1.0 - depth, max(2, i1 - i0))
        env[i0:i1] = np.minimum(env[i0:i1], ramp[: i1 - i0])
        env[i1:id_] = np.minimum(env[i1:id_], 1.0 - depth)
    return np.clip(brightness * env, 0.0, 1.0)


def _style_name(mix_val: float) -> str:
    if mix_val < 0.18:
        return "파도 dimming"
    if mix_val < 0.55:
        return "혼합(파도+둥둥)"
    return "드럼 둥둥"


# ════════════════════════════════════════════════════════════════════
#  재료 계산
# ════════════════════════════════════════════════════════════════════
def _stem_rms_db(stems: dict, raw_wav: Path | None, names=("drums", "bass", "other")):
    """stem 합(없으면 원곡) RMS dB, 0.1s hop. 반환 (rms_db, times) — 오디오 없으면 None."""
    import librosa
    import soundfile as sf

    y = None
    sr = 22050
    parts = [stems.get(n) for n in names]
    parts = [p for p in parts if p is not None and Path(p).exists()]
    if parts:
        sr_in = sr
        for p in parts:
            yi, sr_in = sf.read(str(p), always_2d=True)
            yi = yi.mean(axis=1)
            y = yi if y is None else y[: len(yi)] + yi[: len(y)]
        y = librosa.resample(y.astype(np.float32), orig_sr=sr_in, target_sr=sr)
    elif raw_wav is not None and Path(raw_wav).exists():
        y, sr = librosa.load(str(raw_wav), sr=sr, mono=True)
    else:
        return None

    hop = max(1, int(round(sr * 0.1)))
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-9, ref=np.max(rms) + 1e-9)
    return rms_db, librosa.times_like(rms, sr=sr, hop_length=hop)


def _stem_energy(stems: dict, target_t: np.ndarray, raw_wav: Path | None):
    """반주 stem(drums+bass+other 합) RMS → 0~1 상대 에너지. stem 없으면 원곡으로 폴백."""
    r = _stem_rms_db(stems, raw_wav)
    if r is None:
        return np.full(target_t.size, 0.5)
    rms_db, times = r
    lo, hi = np.percentile(rms_db, 5.0), np.percentile(rms_db, 95.0)
    e = np.clip((rms_db - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    e = _resample_to_grid(e, times, target_t)
    return _gaussian_smooth(e, target_t, ENERGY_SMOOTH_S)


def _label_tracks(segments: list, target_t: np.ndarray):
    """구간 라벨 → 프레임별 (mix 목표, 베이스 레벨 목표) + 후렴 회차 에스컬레이션."""
    mix_t = np.full(target_t.size, PLAN_DEFAULT[0])
    base_t = np.full(target_t.size, PLAN_DEFAULT[1])
    n_chorus = sum(1 for s in segments if s["label"] == "chorus")
    k = 0
    seg_plan = []
    for s in segments:
        m, b = LABEL_PLAN.get(s["label"], PLAN_DEFAULT)
        if s["label"] == "chorus":
            scale = (CHORUS_ESC_START + (1.0 - CHORUS_ESC_START) * (k / (n_chorus - 1))
                     if n_chorus > 1 else 1.0)
            m, b = m * scale, min(b * scale + 0.05, 1.0)
            k += 1
        sel = (target_t >= s["start"]) & (target_t < s["end"])
        mix_t[sel] = m
        base_t[sel] = b
        seg_plan.append((float(s["start"]), float(m)))
    mix_t = _gaussian_smooth(mix_t, target_t, PLAN_RAMP_S)
    base_t = _gaussian_smooth(base_t, target_t, PLAN_RAMP_S)
    return np.clip(mix_t, 0.0, 1.0), np.clip(base_t, 0.0, 1.0), seg_plan


def _beat_flash(beats, downbeats, drums_wav: Path | None, target_t: np.ndarray, bpm: float):
    """allin1 박자 → AR '둥' 그레인(0~1). 세기 = drums 온셋세기 × 박 종류 게인."""
    n = target_t.size
    dt = float(np.median(np.diff(target_t))) if n > 1 else 0.04
    if not beats:
        return np.zeros(n)

    beats = np.asarray(beats, dtype=float)
    down = set(np.round(np.asarray(downbeats, dtype=float) / dt).astype(int).tolist())

    strength = np.ones(beats.size)
    if drums_wav is not None and Path(drums_wav).exists():
        import librosa
        y, sr = librosa.load(str(drums_wav), sr=22050, mono=True)
        hop = max(1, int(round(sr * 0.01)))
        env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        env_t = librosa.times_like(env, sr=sr, hop_length=hop)
        ref = np.percentile(env, DRUM_STR_PCT)
        env_n = np.clip(env / (ref + 1e-8), 0.0, 1.0)
        strength = np.interp(beats, env_t, env_n)
    beats, strength = _min_interval_filter(beats, strength)

    impulse = np.zeros(n)
    for bt, s in zip(beats, strength):
        i = int(round(bt / dt))
        if 0 <= i < n:
            gain = DOWNBEAT_GAIN if i in down else BEAT_GAIN
            impulse[i] = max(impulse[i], float(s) * gain)

    release_s = float(np.clip(FLASH_RELEASE_BEATS * 60.0 / max(bpm, 1.0),
                              FLASH_RELEASE_MIN, FLASH_RELEASE_MAX))
    na = max(1, int(round(FLASH_ATTACK_S / dt)))
    nr = max(2, int(round(release_s / dt)))
    attack = np.linspace(0.0, 1.0, na, endpoint=False)
    release = np.exp(-np.arange(nr) / (nr / 3.0))
    grain = np.concatenate([attack, release])
    out = np.convolve(impulse, grain, mode="full")[:n]
    return np.clip(out, 0.0, 1.0)


# ════════════════════════════════════════════════════════════════════
#  생성 / 저장 / 요약
# ════════════════════════════════════════════════════════════════════
def _resolve_n_frames(segments: list, raw_wav: Path, n_frames: int | None, dt_s: float) -> int:
    if n_frames is not None:
        return n_frames
    dur_s = max((s["end"] for s in segments), default=0.0)
    n_frames = int(round(dur_s / dt_s)) if dur_s > 0 else 0
    if n_frames <= 0:
        import librosa
        n_frames = int(round(librosa.get_duration(path=str(raw_wav)) / dt_s))
    return n_frames


def generate_led(song: str, analysis: dict, raw_wav: Path, *,
                 n_frames: int | None = None, sampling_interval_ms: float = 40.0):
    """LED 밝기 궤적 — LED_MODE 에 따라 분기.

    n_frames 를 주면(입/헤드 행 수) 그 길이로 정확히 맞춘다(프레임 정렬).
    반환: (brightness, summary_rows, mix, bpm)  — final_chorus 모드에서 mix 자리는 info dict.
    """
    if LED_MODE == "final_chorus":
        return generate_led_final_chorus(song, analysis, raw_wav, n_frames=n_frames,
                                         sampling_interval_ms=sampling_interval_ms)
    return generate_led_allin1(song, analysis, raw_wav, n_frames=n_frames,
                               sampling_interval_ms=sampling_interval_ms)


def _final_chorus_span(segments: list):
    """마지막 chorus 와 그 앞에 바로 이어진 chorus 구간들을 합친 (start, end, 구간 수). 없으면 None."""
    idx = [i for i, s in enumerate(segments) if s["label"] == "chorus"]
    if not idx:
        return None
    j = i = idx[-1]
    while i > 0 and segments[i - 1]["label"] == "chorus":
        i -= 1
    return float(segments[i]["start"]), float(segments[j]["end"]), j - i + 1


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _final_chorus_move(analysis: dict, raw_wav: Path | None, t: np.ndarray,
                       c0: float, c1: float) -> np.ndarray:
    """곡 음량(전체 stem 합) → -1~1 움직임. 마지막 후렴 안 중앙값이 0, 분위수 폭으로 정규화, 평활."""
    r = _stem_rms_db(analysis.get("stems", {}), raw_wav, names=("vocals", "drums", "bass", "other"))
    if r is None:
        return np.zeros(t.size)
    rms_db, times = r
    db = _gaussian_smooth(_resample_to_grid(rms_db, times, t), t, FC_MOVE_SMOOTH_S)
    sel = (t >= c0) & (t < c1)
    if not np.any(sel):
        return np.zeros(t.size)
    lo, mid, hi = np.percentile(db[sel], [FC_MOVE_PCT[0], 50.0, FC_MOVE_PCT[1]])
    return np.clip((db - mid) / ((hi - lo) / 2.0 + EPS), -1.0, 1.0)


def generate_led_final_chorus(song: str, analysis: dict, raw_wav: Path, *,
                              n_frames: int | None = None, sampling_interval_ms: float = 40.0):
    """마지막 후렴에서만 점등. 다른 구간은 0, 노래에 반응하는 깜박임 없음."""
    bpm = analysis["bpm"] or 120.0
    segments = analysis["segments"]
    dt_s = sampling_interval_ms / 1000.0
    n_frames = _resolve_n_frames(segments, raw_wav, n_frames, dt_s)
    t = np.arange(n_frames) * dt_s
    p, w, mv = np.zeros(n_frames), np.zeros(n_frames), np.zeros(n_frames)
    song_end = (n_frames - 1) * dt_s          # 마지막 행 = 곡 끝, 여기서 0

    span = _final_chorus_span(segments)
    info = {"span": span, "fade_out": None, "warn": None}
    if span is not None and n_frames > 1:
        c0, c1, _ = span
        c1 = min(c1, song_end)

        # 밝아짐·최대 유지·내려옴 — 후렴이 짧으면 비례 축소
        rise, hold, fall = FC_RISE_S, FC_PEAK_HOLD_S, FC_FALL_S
        intro = rise + hold + fall
        limit = FC_INTRO_MAX_FRAC * max(c1 - c0, 0.0)
        if intro > limit > 0:
            k = limit / intro
            rise, hold, fall = rise * k, hold * k, fall * k
        t_peak, t_fall, t_hold = c0 + rise, c0 + rise + hold, c0 + rise + hold + fall

        # 꺼짐 — 곡 끝에 반드시 0. 여유가 모자라면 줄이고, 너무 짧으면 후렴 끝 전부터 시작
        gap = song_end - c1
        if gap >= FC_FADE_OUT_S:
            fo0, fo1 = c1, c1 + FC_FADE_OUT_S
        elif gap >= FC_FADE_OUT_MIN_S:
            fo0, fo1 = c1, song_end
        else:
            fo0, fo1 = max(t_hold, song_end - FC_FADE_OUT_MIN_S), song_end
            info["warn"] = (f"후렴 끝 여유 {max(gap, 0.0):.1f}s → 꺼짐을 후렴 끝 "
                            f"{c1 - fo0:.1f}s 전부터 시작")
        info["fade_out"] = (fo0, fo1)

        # 성분 분해: b = FC_PEAK·p + w·(FC_HOLD_LEVEL + FC_MOVE_AMP·m)
        #   p = 피크 모양(밝아짐·최대·내려옴), w = 유지 곡선 비중(내려옴·유지·꺼짐), m = 음량 움직임(-1~1)
        # 내려옴은 고정 레벨이 아니라 움직이는 곡선으로 바로 착지 → 착지 후 다시 밝아지는 이음새 없음
        m = (t >= c0) & (t < t_peak)
        p[m] = _smoothstep((t[m] - c0) / max(rise, EPS))
        p[(t >= t_peak) & (t < t_fall)] = 1.0
        m = (t >= t_fall) & (t < t_hold)
        s = _smoothstep((t[m] - t_fall) / max(fall, EPS))
        p[m], w[m] = 1.0 - s, s
        w[(t >= t_hold) & (t < fo1)] = 1.0
        m = t >= fo0
        fade = 1.0 - _smoothstep((t[m] - fo0) / max(fo1 - fo0, EPS))
        p[m] *= fade
        w[m] *= fade
        if FC_MOVE:
            mv = _final_chorus_move(analysis, raw_wav, t, c0, c1)
    if n_frames:
        p[-1] = w[-1] = 0.0
    brightness = np.clip(FC_PEAK * p + w * (FC_HOLD_LEVEL + FC_MOVE_AMP * mv), 0.0, 1.0)
    info["parts"] = (p, w, mv)

    rows = []
    for s in segments:
        i0, i1 = int(s["start"] / dt_s), min(n_frames, int(s["end"] / dt_s))
        if i1 <= i0:
            continue
        in_final = span is not None and span[0] <= s["start"] < span[1]
        rows.append((s["start"], s["end"], s["label"], float(in_final),
                     float(np.min(brightness[i0:i1])), float(np.max(brightness[i0:i1]))))
    return brightness, rows, info, bpm


def generate_led_allin1(song: str, analysis: dict, raw_wav: Path, *,
                        n_frames: int | None = None, sampling_interval_ms: float = 40.0):
    """allin1 설계도 → LED 밝기 궤적 (기존 연출)."""
    bpm = analysis["bpm"] or 120.0
    segments = analysis["segments"]
    dt_s = sampling_interval_ms / 1000.0
    n_frames = _resolve_n_frames(segments, raw_wav, n_frames, dt_s)
    target_t = np.arange(n_frames) * dt_s

    mix, base_label, seg_plan = _label_tracks(segments, target_t)
    energy = _stem_energy(analysis.get("stems", {}), target_t, raw_wav)
    base_level = BASE_W_LABEL * base_label + (1.0 - BASE_W_LABEL) * energy

    base = DIM_FLOOR + (DIM_CEIL - DIM_FLOOR) * base_level
    bar_s = 4.0 * 60.0 / max(bpm, 1.0)
    w = (0.6 * np.sin(2 * np.pi * target_t / (WAVE_BARS1 * bar_s))
         + 0.4 * np.sin(2 * np.pi * target_t / (WAVE_BARS2 * bar_s) + 1.3))
    wave = np.clip(base + WAVE_AMP * w * (1.0 - mix), 0.0, 1.0)

    flash = _beat_flash(analysis["beats"], analysis["downbeats"],
                        analysis.get("stems", {}).get("drums"), target_t, bpm)
    base_eff = wave * (1.0 - DIM_DUCK * mix)
    brightness = np.clip(base_eff + (1.0 - base_eff) * (mix * flash), 0.0, 1.0)
    brightness = np.clip(brightness * BRIGHTNESS_GAIN, 0.0, 1.0)

    drops = [t for (t, m), (_, m_prev) in zip(seg_plan[1:], seg_plan[:-1])
             if m - m_prev >= DROP_JUMP_MIN]
    if drops:
        brightness = apply_drop_anticipation(brightness, drops, bpm, dt_s)

    rows = []
    for s in segments:
        i0, i1 = int(s["start"] / dt_s), min(n_frames, int(s["end"] / dt_s))
        if i1 <= i0:
            continue
        seg_mix = float(np.mean(mix[i0:i1]))
        b0, b1 = float(np.min(brightness[i0:i1])), float(np.max(brightness[i0:i1]))
        rows.append((s["start"], s["end"], s["label"], seg_mix, b0, b1))
    return brightness, rows, mix, bpm


def write_led_csv(out_path: Path, brightness: np.ndarray) -> None:
    """Pi 계약: 1열 밝기, 헤더 없음, 40ms/행. 저장 직전 지각 감마 적용."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    corrected = np.power(np.clip(brightness, 0.0, 1.0), LED_OUT_GAMMA)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for b in corrected:
            w.writerow([f"{b:.4f}"])


def write_led_parts_csv(out_path: Path, parts) -> None:
    """재생 쪽 재조합용 성분(p, w, m). 헤더 1줄 + led.csv 와 같은 행 수, 감마 미적용.
    재생: b = PEAK·p + w·(C + A·m) → b^GAMMA  (기본 PEAK, C, A, GAMMA = FC_PEAK, FC_HOLD_LEVEL, FC_MOVE_AMP, LED_OUT_GAMMA)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["p", "w", "m"])
        for row in zip(*parts):
            wr.writerow([f"{v:.4f}" for v in row])


def summarize(song: str, rows, mix, bpm: float) -> str:
    if LED_MODE == "final_chorus":
        return _summarize_final_chorus(song, rows, mix, bpm)
    out = ["─" * 70]
    out.append(f"  ♪ {song}  [LED: allin1 설계도 — 라벨이 연출 지정]   BPM {bpm:.0f}")
    out.append(f"  주 스타일: {_style_name(float(np.mean(mix)))}")
    out.append("  구간(allin1 라벨 → 연출):")
    for (t0, t1, label, m, b0, b1) in rows:
        ko = _KO.get(label, label)
        out.append(f"    {t0:5.0f}~{t1:<5.0f}s  {ko:5s}  둥둥도 {m:.2f} │ {_style_name(m):14s} 밝기 {b0:.2f}~{b1:.2f}")
    out.append("─" * 70)
    return "\n".join(out)


def _summarize_final_chorus(song: str, rows, info: dict, bpm: float) -> str:
    out = ["─" * 70]
    out.append(f"  ♪ {song}  [LED: 마지막 후렴만 점등]   BPM {bpm:.0f}")
    span = info["span"]
    if span is None:
        out.append("  마지막 후렴: 없음 → 전체 소등")
    else:
        out.append(f"  마지막 후렴: {span[0]:.1f}~{span[1]:.1f}s ({span[2]}개 구간 병합)")
    if info["warn"]:
        out.append(f"  ⚠ {info['warn']}")
    out.append("  구간(allin1 라벨 → 밝기):")
    for (t0, t1, label, in_final, b0, b1) in rows:
        ko = _KO.get(label, label)
        mark = "★" if in_final else " "
        out.append(f"   {mark}{t0:5.0f}~{t1:<5.0f}s  {ko:5s}  밝기 {b0:.2f}~{b1:.2f}")
    out.append("─" * 70)
    return "\n".join(out)


def main():
    utf8_stdout()
    ap = argparse.ArgumentParser(description="[3단계] allin1 설계도 기반 LED 밝기 궤적")
    ap.add_argument("song", help="곡 이름 또는 wav 경로 (분석/<곡>/ 이 먼저 있어야 함)")
    args = ap.parse_args()
    raw = resolve_raw(args.song)
    song = raw.stem

    import p1_allin1_separate as p1
    A = p1.analyze_and_separate(song, raw)
    brightness, rows, mix, bpm = generate_led(song, A, raw)
    out = intermediate_dir(song) / led_csv_name(song)
    write_led_csv(out, brightness)
    print(summarize(song, rows, mix, bpm))
    print(f"  저장: {out}  ({brightness.size} 행)")


if __name__ == "__main__":
    main()
