"""[2단계-a] 비브라토 검출 엔진 — 보컬 wav 에서 '비브라토가 있는 구간(episode)' 을 찾는다.

역할
  음정(F0) 궤적을 뽑고, 느린 음정 변화(중심 음정, center)를 빼서 남은 '잔차(residual, cents)' 에서
  4~8 Hz 로 규칙적으로 흔들리는 부분을 비브라토로 판정한다. 결과(Episode 목록 + 프레임별 특징)는
  p2b_vibrato_motion 이 받아 입 모터 떨림으로 바꾼다.

파이프라인에서 실제로 쓰는 것 (p2b.detect_vibrato_episodes 의 호출 순서)
  AnalysisConfig() ─▶ load_and_preprocess_audio ─▶ extract_pitch_track
                   ─▶ compute_features_analytic_signal ─▶ merge_episodes(METHOD_SPECS["analytic_signal"])
  + p2_offline_motion 이 resolve_audio_path 를 사용.

파일 구성 (위 → 아래)
  §1 설정·데이터 구조 : MethodSpec, AnalysisConfig(★검출 튜닝 파라미터), AudioBundle, PitchTrack, WindowFeatures, Episode
  §2 공용 유틸        : cents 변환, 시간 포맷, NaN-safe 통계, 구멍 메우기
  §3 오디오→음정 궤적 : load_and_preprocess_audio, pyin 디스크 캐시, extract_pitch_track, segment_notes
  §4 프레임별 특징     : compute_features_analytic_signal(★사용) + 3개 비교 방식(standalone 전용, 파이프라인 미사용)
  §5 episode 판정     : build_episode_mask → merge_episodes → summarize_episode
  §6 standalone 출력  : CSV/MAT/MATLAB 스크립트/그림 + CLI(main_for_method)  ← 파이프라인 미사용

용어·단위
  frame  : pyin 분석 단위. hop_length/target_sr 초 간격 (기본 220/22050 ≈ 10 ms → ≈100 fps).
  cents  : 음정 차이 단위. 100 cents = 반음, 1200 cents = 옥타브. 이 파일의 cents 는 C0 기준 절대값.
  rate   : 떨림 속도 [Hz].   extent : 떨림 폭 [cents] — 중심에서 ±(반진폭) 기준.
  episode: 같은 노트(note_id) 안에서 비브라토 조건을 연속으로 만족한 구간.

원본: vibrato_detector.py(= s3a). 로직은 그대로이고 주석만 보강되었다.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np
from scipy.io import savemat
from scipy.signal import butter, filtfilt, find_peaks, hilbert, savgol_filter

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None


# ═══════════════════════════════ §1 설정 · 데이터 구조 ═══════════════════════════════
EPS = 1e-8                    # 0 나눗셈 방지용 아주 작은 수
C0_HZ = 16.351597831287414    # C0 음의 주파수[Hz]. 이 파일의 cents 값은 모두 C0 = 0 cents 기준


@dataclass(frozen=True)
class MethodSpec:
    """검출 방식 하나의 표시용 메타데이터 (출력 파일 접두·그림 제목·축 라벨). 계산 로직은 담지 않는다."""

    method_id: str              # 내부 식별자 (METHOD_SPECS 키 = main_for_method 의 compute_fn 키)
    paper_name: str             # 논문식 방식 이름 (요약 텍스트·그림 제목)
    output_prefix: str          # standalone 출력 파일 이름 접두
    overview_title: str         # 스펙트로그램 그림 제목
    primary_rate_label: str     # WindowFeatures.primary_rate_hz 가 무엇인지 (방식마다 다름)
    secondary_rate_label: str   # WindowFeatures.secondary_rate_hz 가 무엇인지
    metric_a_label: str         # WindowFeatures.metric_a 가 무엇인지
    metric_b_label: str         # WindowFeatures.metric_b 가 무엇인지


@dataclass
class AnalysisConfig:
    """비브라토 '검출' 파라미터 모음 (p2b 의 VibratoMotionConfig 는 '표현' 파라미터).

    파이프라인(p2b.detect_vibrato_episodes)은 항상 기본값 AnalysisConfig() 로 생성한다.
    값을 바꾸려면 여기 기본값을 고치거나, standalone CLI(build_parser) 옵션으로 넘긴다.
    ※ [pyin 캐시 키] 표시가 붙은 필드를 바꾸면 data/pyin_cache 가 무효화되어 pyin 을 다시 돈다(느림).
      나머지 필드는 캐시와 무관하게 매 실행 반영된다.
    """

    # ── (A) 오디오 로드/전처리 ── load_and_preprocess_audio
    target_sr: int = 22050                  # [Hz] 분석 샘플레이트. 로드 시 리샘플                        [pyin 캐시 키]
    normalize_audio: bool = True            # 피크 정규화(최대 절대값 → 1.0) 여부
    use_harmonic_for_pitch: bool = False    # True 면 HPSS 로 하모닉 성분만 pyin 에 넣음 (느림, 기본 끔)

    # ── (B) pyin F0 추출 ── extract_pitch_track → _pyin_cached
    frame_length: int = 2048                # [샘플] pyin 분석 창 길이 (≈93 ms @22050)                     [pyin 캐시 키]
    hop_length: int = 220                   # [샘플] 프레임 간격 (≈10 ms → frame_rate_hz ≈ 100)            [pyin 캐시 키]
    fmin: float = field(default_factory=lambda: float(librosa.note_to_hz("C2")))  # [Hz] F0 탐색 하한 (C2 ≈ 65 Hz)   [pyin 캐시 키]
    fmax: float = field(default_factory=lambda: float(librosa.note_to_hz("C7")))  # [Hz] F0 탐색 상한 (C7 ≈ 2093 Hz) [pyin 캐시 키]

    # ── (C) 중심 음정(center) 추정 → 잔차(residual = F0 − center) ── extract_pitch_track
    flat_window_s: float = 0.45             # [s] 중심 음정용 롤링 중앙값 창. 비브라토 주기(0.12~0.25 s)보다 충분히 길어야 떨림이 잔차에 남음

    # ── (D) 노트 분할 ── segment_notes (episode 는 노트 경계를 넘지 않음)
    note_jump_cents: float = 90.0           # [cents] 인접 유성 프레임의 중심 음정이 이 이상 뛰면 새 노트 (≈반음)
    min_note_duration_s: float = 0.10       # [s] 이보다 짧은 노트는 버림 (note_id = -1)

    # ── (E) 프레임별 특징 계산 창 ── compute_features_*
    method_window_s: float = 0.30           # [s] 프레임마다 rate/extent 를 재는 분석 창 길이 (비브라토 1.5~2 주기)
    search_band_hz: Tuple[float, float] = (3.0, 11.0)   # [Hz] 잔차 대역통과 필터 / 스펙트럼 피크 탐색 범위 (넓게 잡음)

    # ── (F) episode 판정 게이트 ── build_episode_mask / merge_episodes
    detection_band_hz: Tuple[float, float] = (4.0, 8.0)  # [Hz] 최종 rate 가 이 안에 있어야 비브라토 (사람 노래 비브라토 범위)
    min_voiced_ratio: float = 0.80          # [0~1] 분석 창 안 유성(voiced) 프레임 비율 하한 (특징 계산 단계에서도 사용)
    min_extent_cents: float = 10.0          # [cents] 최소 떨림 폭. confidence/score 안의 extent 점수 기준점이기도 함
    min_score: float = 0.45                 # [0~1] window_score 하한 (confidence 하한 0.28 은 build_episode_mask 에 고정)
    min_episode_duration_s: float = 1.50    # [s] 이보다 짧은 연속 구간은 episode 로 인정하지 않음
    merge_gap_s: float = 0.20               # [s] 마스크의 이보다 짧은 끊김은 이어 붙임 (fill_short_false_gaps)
    max_gap_fill_s: float = 0.04            # [s] F0 가 이보다 짧게 끊긴 NaN 구멍은 선형 보간으로 메움 (extract_pitch_track)

    # ── (G) 시각화 ── standalone CLI 전용
    plot_top_k: int = 6                     # 진단 그림에 그릴 상위 episode 개수

    @property
    def frame_rate_hz(self) -> float:
        """[fps] 프레임 시각축의 초당 프레임 수 = target_sr / hop_length (기본 ≈ 100.2)."""
        return self.target_sr / max(self.hop_length, 1)


@dataclass
class AudioBundle:
    """load_and_preprocess_audio 의 결과 — 파형 3종 + 메타."""

    y: np.ndarray        # 원본 파형 (target_sr 로 리샘플, mono, float)
    y_norm: np.ndarray   # 피크 정규화 파형 (normalize_audio=False 면 y 와 동일). 스펙트로그램·MAT 저장용
    y_pitch: np.ndarray  # pyin 입력 파형 (use_harmonic_for_pitch 면 HPSS 하모닉, 아니면 y_norm 그대로)
    sr: int              # [Hz] 실제 샘플레이트 (= target_sr)
    duration: float      # [s] 길이


@dataclass
class PitchTrack:
    """프레임별 음정 궤적 (extract_pitch_track 결과). 모든 배열 길이 = 프레임 수 F."""

    frame_times: np.ndarray      # [s]     각 프레임의 시각 (hop_length/target_sr 간격)
    f0_hz: np.ndarray            # [Hz]    pyin 원본 F0 (무성 프레임 NaN)
    f0_cents: np.ndarray         # [cents] f0_hz 의 cents 변환 (C0 기준, 무성 NaN)
    analysis_cents: np.ndarray   # [cents] f0_cents 에서 짧은 NaN 구멍(≤ max_gap_fill_s)을 보간한 것
    center_cents: np.ndarray     # [cents] 중심 음정 — flat_window_s 롤링 중앙값 + 평균 (멜로디 성분)
    center_hz: np.ndarray        # [Hz]    center_cents 의 Hz 변환
    residual_cents: np.ndarray   # [cents] analysis_cents − center_cents   ★ 비브라토 검출의 입력
    voiced_flag: np.ndarray      # [bool]  pyin 유성 판정
    voiced_prob: np.ndarray      # [0~1]   pyin 유성 확률
    note_id: np.ndarray          # [int]   segment_notes 가 매긴 노트 번호 (무성/짧은 노트 = -1)


@dataclass
class WindowFeatures:
    """프레임별 비브라토 특징 (compute_features_* 결과). 길이 F, 계산되지 않은 프레임은 NaN.

    p2b 가 실제로 쓰는 필드: final_rate_hz, final_extent_cents, confidence (★).
    """

    primary_rate_hz: np.ndarray      # [Hz]    방식별 1차 rate   (analytic_signal: 순시 rate)
    secondary_rate_hz: np.ndarray    # [Hz]    방식별 2차 rate   (analytic_signal: Kay 평활 rate)
    final_rate_hz: np.ndarray        # [Hz]    최종 rate ★       (analytic_signal: = secondary)
    final_extent_cents: np.ndarray   # [cents] 최종 떨림 폭(반진폭) ★
    confidence: np.ndarray           # [0~1]   방식별 가중합 신뢰도 ★
    window_score: np.ndarray         # [0~1]   episode 판정용 점수 (AnalysisConfig.min_score 와 비교)
    voiced_ratio: np.ndarray         # [0~1]   분석 창 안 유성 비율
    agreement: np.ndarray            # [0~1]   1차/2차 rate 일치도 (방식별 정의 다름)
    metric_a: np.ndarray             # 방식별 보조 지표 A (MethodSpec.metric_a_label; analytic_signal: 대역통과 포락선 에너지)
    metric_b: np.ndarray             # 방식별 보조 지표 B (MethodSpec.metric_b_label; analytic_signal: 포락선 표준편차)


@dataclass
class Episode:
    """비브라토 구간 하나 (summarize_episode 결과). 수치는 구간 안 프레임의 window_score 가중 평균.

    p2b 가 실제로 쓰는 필드: start_time, end_time, duration (구간 정의용 ★). 나머지는 요약/그림용.
    """

    note_id: int                 # 구간이 속한 노트 번호 (구간 안 최다 노트)
    start_frame: int             # 시작 프레임 index (포함)
    end_frame: int               # 끝 프레임 index (포함)
    start_time: float            # [s] 시작 시각 = frame_times[start_frame] ★
    end_time: float              # [s] 끝 시각 = frame_times[end_frame] + 1 프레임 간격 ★
    duration: float              # [s] end_time − start_time ★
    mean_f0_hz: float            # [Hz] 구간 평균 F0
    center_pitch_hz: float       # [Hz] 구간 평균 중심 음정
    center_pitch_cents: float    # [cents] 구간 평균 중심 음정
    primary_rate_hz: float       # [Hz] 1차 rate 평균
    secondary_rate_hz: float     # [Hz] 2차 rate 평균
    final_rate_hz: float         # [Hz] 최종 rate 평균
    final_extent_cents: float    # [cents] 최종 extent 평균
    confidence: float            # [0~1] 신뢰도 평균
    window_score: float          # [0~1] 점수 평균
    voiced_ratio: float          # [0~1] 유성 비율 평균
    agreement: float             # [0~1] rate 일치도 평균
    metric_a: float              # 방식별 보조 지표 A 평균
    metric_b: float              # 방식별 보조 지표 B 평균
    final_label: str             # 항상 "existing_vibrato" (기존 비브라토 검출 — 확장 여지용 라벨)
    confidence_tier: str         # "high"(≥0.75) / "medium"(≥0.50) / "low" — summarize_episode 에서 부여
    reason_text: str             # 근거 문장 (build_reason_text)


# 4가지 검출 방식의 표시용 메타데이터. 파이프라인(p2b)은 "analytic_signal" 하나만 쓰고,
# 나머지 셋은 standalone CLI 로 방식을 비교할 때만 사용한다 (main_for_method 의 compute_fn 참조).
METHOD_SPECS: Dict[str, MethodSpec] = {
    "spectrum_modelling": MethodSpec(
        method_id="spectrum_modelling",
        paper_name='Method based on "spectrum modelling"',
        output_prefix="spectrum_modelling",
        overview_title="Spectrum-modelling vibrato detection overview",
        primary_rate_label="Sideband rate",
        secondary_rate_label="Residual FFT rate",
        metric_a_label="Sideband support",
        metric_b_label="Estimated extent",
    ),
    "spectral_envelopes_distortion": MethodSpec(
        method_id="spectral_envelopes_distortion",
        paper_name='Method based on "spectral envelopes distortion"',
        output_prefix="spectral_envelopes_distortion",
        overview_title="Spectral-envelope-distortion vibrato detection overview",
        primary_rate_label="Residual FFT rate",
        secondary_rate_label="Envelope slope rate",
        metric_a_label="Envelope slope",
        metric_b_label="Linearity",
    ),
    "analytic_signal": MethodSpec(
        method_id="analytic_signal",
        paper_name='Method based on the "analytic signal"',
        output_prefix="analytic_signal",
        overview_title="Analytic-signal vibrato detection overview",
        primary_rate_label="Instantaneous rate",
        secondary_rate_label="Kay-smoothed rate",
        metric_a_label="Bandpassed energy",
        metric_b_label="Envelope stability",
    ),
    "minima_maxima_detection": MethodSpec(
        method_id="minima_maxima_detection",
        paper_name='Method based on "minima-maxima detection"',
        output_prefix="minima_maxima_detection",
        overview_title="Minima-maxima vibrato detection overview",
        primary_rate_label="Maxima interval rate",
        secondary_rate_label="Minima interval rate",
        metric_a_label="Interval proportion",
        metric_b_label="Mfreq",
    ),
}


# ═══════════════════════════════ §2 공용 유틸 ═══════════════════════════════
def clamp01(values: np.ndarray | float) -> np.ndarray | float:
    """값을 0~1 로 자른다 (스칼라/배열 모두)."""
    return np.clip(values, 0.0, 1.0)


def hz_to_cents(values: np.ndarray) -> np.ndarray:
    """Hz → cents (C0 기준 절대 cents). 0 이하 값은 NaN."""
    values = np.asarray(values, dtype=float)
    cents = np.full(values.shape, np.nan, dtype=float)
    valid = values > 0
    cents[valid] = 1200.0 * np.log2(values[valid] / C0_HZ)
    return cents


def cents_to_hz(values: np.ndarray) -> np.ndarray:
    """cents (C0 기준) → Hz. NaN/inf 는 NaN."""
    values = np.asarray(values, dtype=float)
    hz = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    hz[valid] = C0_HZ * np.power(2.0, values[valid] / 1200.0)
    return hz


def seconds_to_timestamp(seconds: float) -> str:
    """초 → 'MM:SS.mmm' 문자열 (요약 텍스트·그림 제목용)."""
    total_ms = int(round(max(0.0, seconds) * 1000.0))
    minutes, ms_remainder = divmod(total_ms, 60000)
    secs, millis = divmod(ms_remainder, 1000)
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


def log_progress(message: str) -> None:
    """[HH:MM:SS] 접두를 붙여 즉시 출력 (standalone CLI 진행 로그)."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def format_elapsed(seconds: float) -> str:
    """경과 시간(초) → '12.34s' 또는 '1m 2.34s'."""
    seconds = float(max(0.0, seconds))
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60.0)
    remain = seconds - 60.0 * minutes
    return f"{minutes}m {remain:.2f}s"


def create_run_output_dir(base_dir: Path, audio_path: Path) -> Path:
    """base_dir/<오디오 stem>_<YYYYmmdd_HHMMSS>[_NN]/ 폴더를 새로 만들어 반환 (standalone CLI 출력용)."""
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = audio_path.stem.replace(" ", "_")
    candidate = base_dir / f"{stem}_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = base_dir / f"{stem}_{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def resolve_audio_path(audio_path_arg: str) -> Path:
    """오디오 인자 → 실제 파일 경로.

    그대로 존재하면 그 경로를 쓰고(파이프라인은 항상 이 경우), 아니면 레거시 위치인
    music/vocals/, music/ 을 현재 작업 폴더와 이 모듈 폴더 기준으로 차례로 찾는다.
    못 찾으면 원래 인자를 Path 로 그대로 돌려준다(존재 여부는 호출 측이 처리).
    """
    candidate = Path(audio_path_arg)
    if candidate.exists():
        return candidate
    # 레거시 배치: 보컬 wav 가 music/vocals/ (또는 music/) 에 있던 시절의 탐색 규칙.
    # 어디서 스크립트를 실행해도 찾을 수 있게 현재 작업 폴더와 이 모듈 폴더 둘 다 본다.
    # (현재 파이프라인은 분석/<곡>/<곡>_vocals.wav 의 절대 경로를 넘기므로 위 candidate.exists() 에서 끝난다)
    script_root = Path(__file__).resolve().parent
    search_dirs = [
        Path("music") / "vocals",
        Path("music"),
        script_root / "music" / "vocals",
        script_root / "music",
    ]
    for base in search_dirs:
        music_candidate = base / audio_path_arg
        if music_candidate.exists():
            return music_candidate
    return candidate


def safe_nanmean(values: np.ndarray, default: float = np.nan) -> float:
    """NaN 을 무시한 평균. 유효값이 하나도 없으면 default."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return float(default)
    return float(np.nanmean(arr))


def weighted_nanmean(values: np.ndarray, weights: np.ndarray, default: float = np.nan) -> float:
    """가중 평균 (NaN 값·NaN/비양수 가중치 항은 제외). 유효 항이 없으면 default."""
    arr = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    valid = np.isfinite(arr) & np.isfinite(w) & (w > 0)
    if not np.any(valid):
        return float(default)
    return float(np.sum(arr[valid] * w[valid]) / np.sum(w[valid]))


def rolling_nanmedian(values: np.ndarray, window: int) -> np.ndarray:
    """창 크기 window(프레임) 의 중심 롤링 NaN-무시 중앙값. 창 안이 전부 NaN 이면 NaN. (순수 파이썬 루프 — 프레임 수 × 창 크기)"""
    arr = np.asarray(values, dtype=float)
    result = np.full(arr.shape, np.nan, dtype=float)
    half = max(1, window // 2)
    for idx in range(len(arr)):
        chunk = arr[max(0, idx - half) : min(len(arr), idx + half + 1)]
        if np.any(np.isfinite(chunk)):
            result[idx] = np.nanmedian(chunk)
    return result


def rolling_nanmean(values: np.ndarray, window: int) -> np.ndarray:
    """창 크기 window(프레임) 의 중심 롤링 NaN-무시 평균 (rolling_nanmedian 과 같은 구조)."""
    arr = np.asarray(values, dtype=float)
    result = np.full(arr.shape, np.nan, dtype=float)
    half = max(1, window // 2)
    for idx in range(len(arr)):
        chunk = arr[max(0, idx - half) : min(len(arr), idx + half + 1)]
        if np.any(np.isfinite(chunk)):
            result[idx] = np.nanmean(chunk)
    return result


def fill_short_nan_gaps(values: np.ndarray, max_gap: int) -> np.ndarray:
    """길이 ≤ max_gap(프레임) 인 NaN 구멍을 양 옆 유효값으로 선형 보간. 양 끝에 걸린 구멍은 그대로 둔다."""
    arr = np.asarray(values, dtype=float).copy()
    idx = 0
    while idx < len(arr):
        if np.isfinite(arr[idx]):
            idx += 1
            continue
        start = idx
        while idx < len(arr) and not np.isfinite(arr[idx]):
            idx += 1
        end = idx
        gap = end - start
        left = start - 1
        right = end
        if gap <= max_gap and left >= 0 and right < len(arr) and np.isfinite(arr[left]) and np.isfinite(arr[right]):
            arr[start:end] = np.linspace(arr[left], arr[right], gap + 2)[1:-1]
    return arr


def fill_short_false_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """bool 마스크에서 True 사이에 끼인 길이 ≤ max_gap(프레임) 의 False 구간을 True 로 메움. (p2b._merge_short_gaps 의 프레임 단위 버전)"""
    arr = np.asarray(mask, dtype=bool).copy()
    idx = 0
    while idx < len(arr):
        if arr[idx]:
            idx += 1
            continue
        start = idx
        while idx < len(arr) and not arr[idx]:
            idx += 1
        end = idx
        if (end - start) <= max_gap and start > 0 and end < len(arr) and arr[start - 1] and arr[end]:
            arr[start:end] = True
    return arr


def matlab_cell_column(values: Sequence[str]) -> np.ndarray:
    """문자열 목록 → MATLAB cell 열 벡터(object 배열, N×1) — savemat 용."""
    return np.asarray(list(values), dtype=object).reshape(-1, 1)


# ═══════════════════════════════ §3 오디오 → 음정 궤적 ═══════════════════════════════
def load_and_preprocess_audio(audio_path: Path, config: AnalysisConfig) -> AudioBundle:
    """wav 로드(target_sr, mono) → 피크 정규화 → (옵션) HPSS 하모닉 추출 → AudioBundle.

    y       : 원본 파형(리샘플만)            y_norm : 피크 정규화본 (스펙트로그램·MAT 저장용)
    y_pitch : pyin 입력. use_harmonic_for_pitch 면 하모닉 성분, 아니면 y_norm 그대로.
    """
    y, sr = librosa.load(audio_path, sr=config.target_sr, mono=True)
    y = np.asarray(y, dtype=float)
    y_norm = y.copy()
    peak = np.max(np.abs(y_norm)) if y_norm.size else 0.0
    if config.normalize_audio and peak > EPS:
        y_norm = y_norm / peak
    if config.use_harmonic_for_pitch:
        y_pitch, _ = librosa.effects.hpss(y_norm)
    else:
        y_pitch = y_norm
    return AudioBundle(y=y, y_norm=y_norm, y_pitch=y_pitch, sr=sr, duration=float(len(y) / max(sr, 1)))


def segment_notes(center_cents: np.ndarray, voiced_flag: np.ndarray, frame_times: np.ndarray, config: AnalysisConfig) -> np.ndarray:
    """중심 음정(center_cents) 이 note_jump_cents 이상 뛰는 지점에서 노트를 나눠 프레임별 note_id 를 만든다.

    - 무성(voiced_flag=False) 또는 center 가 NaN 인 프레임은 -1 이고, 그 다음 유성 프레임에서 새 노트가 시작된다.
    - 길이 < min_note_duration_s 인 노트는 통째로 -1 로 버리고, 남은 id 를 0,1,2,… 로 다시 매긴다.
    episode 는 노트 경계를 넘지 않으므로(merge_episodes) 이 분할이 episode 의 최대 길이를 정한다.
    """
    note_id = np.full(len(center_cents), -1, dtype=int)
    current_id = -1
    last_idx: Optional[int] = None
    for idx in range(len(center_cents)):
        if not voiced_flag[idx] or not np.isfinite(center_cents[idx]):
            last_idx = None
            continue
        # 직전 유성 프레임과 중심 음정 차이가 note_jump_cents 이상이면 새 노트 시작
        if last_idx is None or abs(center_cents[idx] - center_cents[last_idx]) >= config.note_jump_cents:
            current_id += 1
        note_id[idx] = current_id
        last_idx = idx

    frame_step = float(np.nanmedian(np.diff(frame_times))) if len(frame_times) > 1 else 0.0
    # 너무 짧은 노트 제거 → 남은 노트 id 를 0 부터 연속으로 재번호
    min_frames = max(1, int(round(config.min_note_duration_s / max(frame_step, EPS))))
    for nid in sorted(x for x in np.unique(note_id) if x >= 0):
        indices = np.flatnonzero(note_id == nid)
        if len(indices) < min_frames:
            note_id[indices] = -1
    unique_ids = sorted(x for x in np.unique(note_id) if x >= 0)
    remap = {old: new for new, old in enumerate(unique_ids)}
    for old, new in remap.items():
        note_id[note_id == old] = new
    return note_id


# pyin 결과 디스크 캐시 폴더 = <이 파일 폴더>/data/pyin_cache
# (common.DATA_DIR / "pyin_cache" 와 같은 위치지만 common 을 import 하지 않고 따로 정의되어 있음)
_PYIN_CACHE_DIR = Path(__file__).resolve().parent / "data" / "pyin_cache"


def _pyin_cache_key(y_pitch, sr, frame_length, hop_length, fmin, fmax) -> str:
    """pyin 입력(오디오 배열 + 파라미터) 의 blake2b 해시 → 캐시 파일 이름."""
    # 오디오 배열 자체와 pyin 파라미터를 함께 해시 → 오디오·전처리·pyin 설정 중 하나라도 바뀌면
    # 키가 달라져 자동으로 다시 계산된다. 끝의 'pyin_v1' 은 캐시 포맷 버전 표시.
    h = hashlib.blake2b(digest_size=16)
    h.update(np.ascontiguousarray(y_pitch, dtype=np.float64).tobytes())
    h.update(f"{sr}|{frame_length}|{hop_length}|{fmin:.6f}|{fmax:.6f}|pyin_v1".encode())
    return h.hexdigest()


def _pyin_cached(y_pitch, sr, frame_length, hop_length, fmin, fmax):
    """librosa.pyin 을 디스크(data/pyin_cache/<키>.npz)에 메모이즈한 버전.

    전체 실행 시간의 대부분이 pyin(비터비 음정 추적)인데, 그 결과는 캐시 키에 들어가는
    오디오·pyin 파라미터에만 의존한다. 그 뒤 단계(노트 분할·특징·episode)는 매 실행
    AnalysisConfig 로 다시 계산하므로 검출 파라미터 튜닝은 캐시와 무관하다.
    캐시 미스나 읽기/쓰기 오류는 조용히 일반 pyin 호출로 대체된다(파이프라인을 깨지 않음).
    반환: (f0_hz, voiced_flag, voiced_prob) — librosa.pyin 과 동일.
    """
    key = _pyin_cache_key(y_pitch, sr, frame_length, hop_length, fmin, fmax)
    cache_path = _PYIN_CACHE_DIR / f"{key}.npz"
    try:
        with np.load(cache_path) as data:
            print(f"[pyin-cache] hit: {cache_path.name}")
            return data["f0_hz"], data["voiced_flag"], data["voiced_prob"]
    except Exception:
        pass  # 캐시 미스 또는 손상 → 다시 계산

    f0_hz, voiced_flag, voiced_prob = librosa.pyin(
        y_pitch,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
    )

    try:
        _PYIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(cache_path.name + ".tmp")
        with open(tmp_path, "wb") as fp:  # 파일 핸들로 넘기면 np.savez 가 .npz 확장자를 덧붙이지 않음
            np.savez(fp, f0_hz=f0_hz, voiced_flag=voiced_flag, voiced_prob=voiced_prob)
        tmp_path.replace(cache_path)  # 임시 파일 → 원자적 교체 (쓰다 만 캐시가 남지 않게)
        print(f"[pyin-cache] miss -> computed & cached: {cache_path.name}")
    except Exception:
        pass  # 캐시 저장은 최선 노력 — 실패해도 파이프라인은 계속

    return f0_hz, voiced_flag, voiced_prob


def extract_pitch_track(audio: AudioBundle, config: AnalysisConfig) -> PitchTrack:
    """pyin F0 → cents → 짧은 구멍 메우기 → 중심 음정(롤링 중앙값+평균) → 잔차 → 노트 분할 → PitchTrack.

    잔차(residual_cents = analysis_cents − center_cents) 가 비브라토 검출의 핵심 입력이다:
    느린 멜로디 움직임은 center 로 빠지고 4~8 Hz 떨림만 잔차에 남도록 flat_window_s 가 창 길이를 정한다.
    """
    f0_hz, voiced_flag, voiced_prob = _pyin_cached(
        audio.y_pitch,
        audio.sr,
        config.frame_length,
        config.hop_length,
        config.fmin,
        config.fmax,
    )
    frame_times = librosa.times_like(f0_hz, sr=audio.sr, hop_length=config.hop_length)  # [s] 각 프레임의 시각
    f0_hz = np.asarray(f0_hz, dtype=float)
    voiced_flag = np.asarray(voiced_flag, dtype=bool)
    voiced_prob = np.asarray(voiced_prob, dtype=float)
    f0_cents = hz_to_cents(f0_hz)
    gap_frames = max(1, int(round(config.max_gap_fill_s * config.frame_rate_hz)))  # 메울 수 있는 짧은 무성 구멍 길이(프레임)
    analysis_cents = fill_short_nan_gaps(f0_cents, gap_frames)
    flat_window = max(3, int(round(config.flat_window_s * config.frame_rate_hz)))  # 중심 음정 창(프레임, 최소 3)
    # 중심 음정: 롤링 중앙값(떨림에 강건) → 절반 창 롤링 평균(계단 완화)
    center_cents = rolling_nanmedian(analysis_cents, flat_window)
    center_cents = rolling_nanmean(center_cents, max(3, flat_window // 2))
    center_hz = cents_to_hz(center_cents)
    residual_cents = analysis_cents - center_cents  # [cents] 비브라토 성분이 남는 잔차 ★
    note_id = segment_notes(center_cents, voiced_flag, frame_times, config)
    return PitchTrack(
        frame_times=frame_times,
        f0_hz=f0_hz,
        f0_cents=f0_cents,
        analysis_cents=analysis_cents,
        center_cents=center_cents,
        center_hz=center_hz,
        residual_cents=residual_cents,
        voiced_flag=voiced_flag,
        voiced_prob=voiced_prob,
        note_id=note_id,
    )


# ═══════════════════════════════ §4 프레임별 특징 계산 ═══════════════════════════════
# 파이프라인은 compute_features_analytic_signal 만 사용한다. 나머지 세 방식은 standalone 비교용.
def spectral_peak_rate(signal_values: np.ndarray, frame_rate_hz: float, search_band_hz: Tuple[float, float]) -> Tuple[float, float]:
    """잔차 조각의 FFT 에서 search_band_hz 안 최대 피크 → (피크 주파수[Hz], 피크 크기).

    평균 제거 · Hanning 창 · 2의 거듭제곱 zero-pad(≥256) 후 rfft. 유효값 4개 미만이면 (nan, nan).
    analytic_signal 방식은 직접 쓰지 않고, 다른 방식들의 참조 rate(compute_reference_rate_and_extent)에 쓰인다.
    """
    x = np.asarray(signal_values, dtype=float)
    if len(x) < 4 or not np.any(np.isfinite(x)):
        return np.nan, np.nan
    x = x - np.nanmean(x)
    x = np.nan_to_num(x, nan=0.0)
    windowed = x * np.hanning(len(x))
    n_fft = max(256, int(2 ** math.ceil(math.log2(max(len(x), 16)))))
    mag = np.abs(np.fft.rfft(windowed, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / frame_rate_hz)
    band = (freqs >= search_band_hz[0]) & (freqs <= search_band_hz[1])
    if not np.any(band):
        return np.nan, np.nan
    band_mag = mag[band]
    band_freqs = freqs[band]
    peak_idx = int(np.argmax(band_mag))
    return float(band_freqs[peak_idx]), float(band_mag[peak_idx])


def slice_for_index(index: int, total: int, window_frames: int) -> Optional[slice]:
    """index 를 중심으로 window_frames 길이의 slice. 배열 경계를 벗어나면 None (그 프레임은 특징 계산 생략)."""
    half = window_frames // 2
    start = index - half
    end = start + window_frames
    if start < 0 or end > total:
        return None
    return slice(start, end)


def frame_center_samples(index: int, config: AnalysisConfig) -> Tuple[int, int]:
    """프레임 index 의 중심 샘플 기준 [start, end) 샘플 범위. (현재 어디서도 호출되지 않는 레거시 함수)"""
    center = index * config.hop_length
    half = config.frame_length // 2
    return center - half, center + half


def extract_audio_window(audio: AudioBundle, index: int, size_samples: int, config: AnalysisConfig) -> Optional[np.ndarray]:
    """프레임 index 를 중심으로 size_samples 길이의 파형(y_pitch) 조각. 경계 밖이면 None. (spectrum/envelope 방식 전용)"""
    center = index * config.hop_length
    half = size_samples // 2
    start = center - half
    end = start + size_samples
    if start < 0 or end > len(audio.y_pitch):
        return None
    return np.asarray(audio.y_pitch[start:end], dtype=float)


def build_empty_features(frame_count: int) -> WindowFeatures:
    """모든 필드가 NaN(길이 frame_count) 인 WindowFeatures. compute_features_* 가 계산된 프레임만 채운다."""
    template = np.full(frame_count, np.nan, dtype=float)
    return WindowFeatures(
        primary_rate_hz=template.copy(),
        secondary_rate_hz=template.copy(),
        final_rate_hz=template.copy(),
        final_extent_cents=template.copy(),
        confidence=template.copy(),
        window_score=template.copy(),
        voiced_ratio=template.copy(),
        agreement=template.copy(),
        metric_a=template.copy(),
        metric_b=template.copy(),
    )


def compute_reference_rate_and_extent(track: PitchTrack, seg: slice, config: AnalysisConfig) -> Tuple[float, float]:
    """잔차 조각(seg) 의 스펙트럼 피크 rate[Hz] 와 extent[cents] = (95백분위 − 5백분위)/2. 다른 방식들의 참조값."""
    residual = track.residual_cents[seg]
    rate, _ = spectral_peak_rate(residual, config.frame_rate_hz, config.search_band_hz)
    extent = 0.5 * (np.nanpercentile(residual, 95.0) - np.nanpercentile(residual, 5.0)) if np.any(np.isfinite(residual)) else np.nan
    return float(rate), float(extent)


def compute_features_analytic_signal(audio: AudioBundle, track: PitchTrack, config: AnalysisConfig) -> WindowFeatures:
    """★ 파이프라인이 쓰는 특징 계산 — '해석 신호(analytic signal)' 방식.

    전역 단계 (한 번):
      잔차(cents, NaN→0) → 2차 Butterworth 대역통과(search_band_hz) → 힐베르트 변환
      → 순시 위상 미분 = 순시 rate[Hz] (primary), Savitzky-Golay(0.35 s 창) 평활 rate (secondary = final)
      → |해석 신호| = 떨림 포락선 진폭
    프레임 단계 (유성 & 노트 안 프레임마다, method_window_s 창):
      창의 70% 이상이 같은 노트이고 유성 비율 ≥ min_voiced_ratio 일 때만 계산.
      extent       = 대역통과 잔차의 (95백분위 − 5백분위)/2   [cents]
      agreement    = 순시 rate 와 평활 rate 의 일치도 (차이 3 Hz 에서 0)
      confidence   = 0.35·agreement + 0.30·진폭점수((amp−2)/20) + 0.20·유성비율 + 0.15·extent점수((extent−min)/20)
      window_score = 0.60·confidence + 0.25·extent점수'((extent−min)/15) + 0.15·유성비율
      metric_a     = 포락선 평균(대역통과 에너지),  metric_b = 포락선 표준편차(안정성)
    ※ 가중치·0.70·0.35 s·(2.0, 20.0, 15.0) 등은 코드에 고정된 상수다 (AnalysisConfig 에 없음).
    """
    features = build_empty_features(len(track.frame_times))
    residual = np.asarray(track.residual_cents, dtype=float)
    # 무성 구간(NaN)은 0 으로 채워 필터가 끊기지 않게
    filled = np.nan_to_num(residual, nan=0.0)
    # 프레임 시각축 기준 2차 Butterworth 대역통과 (search_band_hz, 기본 3~11 Hz). 32 프레임 이하면 필터 생략
    nyquist = 0.5 * config.frame_rate_hz
    b, a = butter(2, [config.search_band_hz[0] / nyquist, config.search_band_hz[1] / nyquist], btype="bandpass")
    filtered = filtfilt(b, a, filled) if len(filled) > 32 else filled
    # 해석 신호: 위상 → 순시 주파수(rate), 크기 → 떨림 포락선(amplitude)
    analytic = hilbert(filtered)
    inst_phase = np.unwrap(np.angle(analytic))
    inst_rate = np.gradient(inst_phase) * config.frame_rate_hz / (2.0 * np.pi)  # [Hz] 순시 떨림 속도
    # Kay 방식 평활: 0.35 s 홀수 창 Savitzky-Golay(2차) 로 순시 rate 의 잡음 억제
    kay_window = max(5, int(round(0.35 * config.frame_rate_hz)))
    if kay_window % 2 == 0:
        kay_window += 1
    kay_rate = savgol_filter(inst_rate, kay_window, polyorder=2, mode="interp") if len(inst_rate) > kay_window else inst_rate
    amplitude = np.abs(analytic)  # 떨림 포락선 진폭 (대역통과 잔차 기준, cents 스케일)

    for idx in range(len(track.frame_times)):
        if not track.voiced_flag[idx] or track.note_id[idx] < 0:
            continue
        window_frames = max(8, int(round(config.method_window_s * config.frame_rate_hz)))
        seg = slice_for_index(idx, len(track.frame_times), window_frames)
        if seg is None:
            continue
        # 창의 70% 이상이 현재 프레임과 같은 노트여야 함 (노트 경계에 걸친 창 제외)
        note_slice = track.note_id[seg]
        if np.mean(note_slice == track.note_id[idx]) < 0.70:
            continue
        voiced_ratio = float(np.mean(track.voiced_flag[seg]))
        if voiced_ratio < config.min_voiced_ratio:
            continue
        # 창 평균 rate 두 종류 + extent + 포락선 통계
        local_rate = safe_nanmean(inst_rate[seg])
        smooth_rate = safe_nanmean(kay_rate[seg])
        extent = 0.5 * (np.nanpercentile(filtered[seg], 95.0) - np.nanpercentile(filtered[seg], 5.0))
        amp = safe_nanmean(amplitude[seg])
        env_std = float(np.nanstd(amplitude[seg]))
        # 점수화 (공식은 함수 docstring 참조)
        agreement = float(clamp01(1.0 - abs(local_rate - smooth_rate) / 3.0))
        confidence = float(clamp01(0.35 * agreement + 0.30 * clamp01((amp - 2.0) / 20.0) + 0.20 * voiced_ratio + 0.15 * clamp01((extent - config.min_extent_cents) / 20.0)))
        score = float(clamp01(0.6 * confidence + 0.25 * clamp01((extent - config.min_extent_cents) / 15.0) + 0.15 * voiced_ratio))

        features.primary_rate_hz[idx] = local_rate
        features.secondary_rate_hz[idx] = smooth_rate
        features.final_rate_hz[idx] = smooth_rate
        features.final_extent_cents[idx] = extent
        features.confidence[idx] = confidence
        features.window_score[idx] = score
        features.voiced_ratio[idx] = voiced_ratio
        features.agreement[idx] = agreement
        features.metric_a[idx] = amp
        features.metric_b[idx] = env_std
    return features


def compute_features_minima_maxima_detection(audio: AudioBundle, track: PitchTrack, config: AnalysisConfig) -> WindowFeatures:
    """[standalone 비교용 — 파이프라인 미사용] '극대·극소 검출' 방식.

    평활 잔차의 봉우리/골 간격으로 rate(= 1/평균 간격), 간격 분산(안정성), 0.15~0.25 s 간격 비율(proportion),
    Mfreq(봉우리−골 상대 크기) 를 계산해 confidence/score 를 만든다.
    """
    features = build_empty_features(len(track.frame_times))
    residual = np.asarray(track.residual_cents, dtype=float)
    if len(residual) < 5:
        return features
    smooth = savgol_filter(np.nan_to_num(residual, nan=0.0), 7, 2, mode="interp") if len(residual) >= 7 else np.nan_to_num(residual, nan=0.0)
    maxima, _ = find_peaks(smooth, distance=max(2, int(round(0.12 * config.frame_rate_hz))))
    minima, _ = find_peaks(-smooth, distance=max(2, int(round(0.12 * config.frame_rate_hz))))

    for idx in range(len(track.frame_times)):
        if not track.voiced_flag[idx] or track.note_id[idx] < 0:
            continue
        window_frames = max(8, int(round(config.method_window_s * config.frame_rate_hz)))
        seg = slice_for_index(idx, len(track.frame_times), window_frames)
        if seg is None:
            continue
        local_max = maxima[(maxima >= seg.start) & (maxima < seg.stop)]
        local_min = minima[(minima >= seg.start) & (minima < seg.stop)]
        if len(local_max) < 2 or len(local_min) < 2:
            continue
        max_intervals = np.diff(track.frame_times[local_max])
        min_intervals = np.diff(track.frame_times[local_min])
        vmax = float(np.var(max_intervals)) if len(max_intervals) else np.nan
        vmin = float(np.var(min_intervals)) if len(min_intervals) else np.nan
        pmax = float(np.mean((max_intervals >= 0.15) & (max_intervals <= 0.25))) if len(max_intervals) else np.nan
        pmin = float(np.mean((min_intervals >= 0.15) & (min_intervals <= 0.25))) if len(min_intervals) else np.nan
        max_vals = smooth[local_max[: min(len(local_max), len(local_min))]]
        min_vals = smooth[local_min[: min(len(local_max), len(local_min))]]
        mfreq = safe_nanmean((max_vals - min_vals) / np.maximum((max_vals + min_vals) * 0.5 + 1e-3, 1e-3), default=np.nan)
        max_rate = 1.0 / safe_nanmean(max_intervals, default=np.nan)
        min_rate = 1.0 / safe_nanmean(min_intervals, default=np.nan)
        extent = 0.5 * safe_nanmean(max_vals - min_vals, default=np.nan)
        voiced_ratio = float(np.mean(track.voiced_flag[seg]))
        agreement = float(clamp01(1.0 - abs(max_rate - min_rate) / 2.0)) if np.isfinite(max_rate) and np.isfinite(min_rate) else 0.0
        stability = float(clamp01(1.0 - (np.nan_to_num(vmax, nan=1.0) + np.nan_to_num(vmin, nan=1.0)) / 0.01))
        proportion = float(np.nanmean([pmax, pmin]))
        confidence = float(clamp01(0.30 * proportion + 0.25 * stability + 0.20 * agreement + 0.15 * clamp01((np.nan_to_num(mfreq, nan=0.0) - 0.03) / 0.08) + 0.10 * voiced_ratio))
        score = float(clamp01(0.65 * confidence + 0.20 * clamp01((extent - config.min_extent_cents) / 20.0) + 0.15 * voiced_ratio))

        features.primary_rate_hz[idx] = max_rate
        features.secondary_rate_hz[idx] = min_rate
        features.final_rate_hz[idx] = safe_nanmean(np.asarray([max_rate, min_rate], dtype=float), default=np.nan)
        features.final_extent_cents[idx] = extent
        features.confidence[idx] = confidence
        features.window_score[idx] = score
        features.voiced_ratio[idx] = voiced_ratio
        features.agreement[idx] = agreement
        features.metric_a[idx] = proportion
        features.metric_b[idx] = mfreq
    return features


def compute_features_spectral_envelopes_distortion(audio: AudioBundle, track: PitchTrack, config: AnalysisConfig) -> WindowFeatures:
    """[standalone 비교용 — 파이프라인 미사용] '스펙트럼 포락선 왜곡' 방식.

    20 ms / 60 ms 두 창의 스펙트럼 포락선 차이의 주파수 기울기(slope)와 선형성(linearity)으로 비브라토 여부를 점수화.
    rate/extent 자체는 잔차 FFT 참조값(compute_reference_rate_and_extent)을 그대로 쓴다. 매 프레임 파형 FFT 라 느림.
    """
    features = build_empty_features(len(track.frame_times))
    short_size = max(256, int(round(0.020 * audio.sr)))
    long_size = max(short_size + 2, int(round(0.060 * audio.sr)))
    freq_limit_hz = min(5000.0, 0.5 * audio.sr)

    for idx in range(len(track.frame_times)):
        if not track.voiced_flag[idx] or track.note_id[idx] < 0:
            continue
        short_win = extract_audio_window(audio, idx, short_size, config)
        long_win = extract_audio_window(audio, idx, long_size, config)
        if short_win is None or long_win is None:
            continue
        short_mag = np.abs(np.fft.rfft(short_win * np.hanning(len(short_win)), n=2048))
        long_mag = np.abs(np.fft.rfft(long_win * np.hanning(len(long_win)), n=2048))
        freqs = np.fft.rfftfreq(2048, d=1.0 / audio.sr)
        band = (freqs >= max(track.center_hz[idx], 80.0)) & (freqs <= freq_limit_hz)
        if np.count_nonzero(band) < 20:
            continue
        env_short = savgol_filter(short_mag, 31, 2, mode="interp")
        env_long = savgol_filter(long_mag, 31, 2, mode="interp")
        d = (env_short[band] - env_long[band]) / np.maximum(env_short[band], EPS)
        coeffs = np.polyfit(freqs[band], d, deg=1)
        slope = float(coeffs[0])
        trend = np.polyval(coeffs, freqs[band])
        ss_res = float(np.sum((d - trend) ** 2))
        ss_tot = float(np.sum((d - np.mean(d)) ** 2))
        linearity = float(1.0 - ss_res / max(ss_tot, EPS))
        voiced_ratio = float(track.voiced_flag[idx])
        ref_seg = slice_for_index(idx, len(track.frame_times), max(8, int(round(config.method_window_s * config.frame_rate_hz))))
        if ref_seg is None:
            continue
        residual_rate, residual_extent = compute_reference_rate_and_extent(track, ref_seg, config)
        slope_score = float(clamp01((slope - 1e-7) / 8e-7))
        confidence = float(clamp01(0.45 * slope_score + 0.30 * clamp01(linearity) + 0.15 * voiced_ratio + 0.10 * clamp01((np.nan_to_num(residual_extent, nan=0.0) - config.min_extent_cents) / 20.0)))
        score = float(clamp01(0.65 * confidence + 0.20 * clamp01(linearity) + 0.15 * clamp01((np.nan_to_num(residual_extent, nan=0.0) - config.min_extent_cents) / 12.0)))

        features.primary_rate_hz[idx] = residual_rate
        features.secondary_rate_hz[idx] = residual_rate
        features.final_rate_hz[idx] = residual_rate
        features.final_extent_cents[idx] = residual_extent
        features.confidence[idx] = confidence
        features.window_score[idx] = score
        features.voiced_ratio[idx] = voiced_ratio
        features.agreement[idx] = clamp01(linearity)
        features.metric_a[idx] = slope
        features.metric_b[idx] = linearity
    return features


def compute_features_spectrum_modelling(audio: AudioBundle, track: PitchTrack, config: AnalysisConfig) -> WindowFeatures:
    """[standalone 비교용 — 파이프라인 미사용] '스펙트럼 모델링' 방식.

    각 하모닉 주변의 ±rate 측대파(sideband) 에너지가 rate×extent 후보 격자(33×19) 중 어디서 가장 잘 맞는지 탐색.
    프레임마다 3중 루프라 네 방식 중 가장 느리다.
    """
    features = build_empty_features(len(track.frame_times))
    size_samples = max(int(round(0.040 * audio.sr)), 512)
    rate_candidates = np.linspace(config.search_band_hz[0], config.search_band_hz[1], 33)
    extent_candidates = np.linspace(8.0, 80.0, 19)

    for idx in range(len(track.frame_times)):
        f0 = track.center_hz[idx]
        if not track.voiced_flag[idx] or track.note_id[idx] < 0 or not np.isfinite(f0) or f0 <= 0:
            continue
        window = extract_audio_window(audio, idx, size_samples, config)
        if window is None:
            continue
        spectrum = np.abs(np.fft.rfft(window * np.hanning(len(window)), n=4096))
        freqs = np.fft.rfftfreq(4096, d=1.0 / audio.sr)
        harmonic_count = max(1, min(4, int((0.5 * audio.sr) / max(f0, EPS))))
        best_score = -np.inf
        best_rate = np.nan
        best_extent = np.nan

        for rate in rate_candidates:
            for extent_cents in extent_candidates:
                delta_hz = f0 * (2.0 ** (extent_cents / 1200.0) - 1.0)
                support = 0.0
                center_energy = 0.0
                for harmonic in range(1, harmonic_count + 1):
                    center_freq = harmonic * f0
                    left_freq = center_freq - rate
                    right_freq = center_freq + rate
                    if left_freq <= 0 or right_freq >= freqs[-1]:
                        continue
                    center_energy += float(np.interp(center_freq, freqs, spectrum))
                    support += float(np.interp(left_freq, freqs, spectrum) + np.interp(right_freq, freqs, spectrum))
                    support += 0.10 * delta_hz
                ratio = support / max(center_energy, EPS)
                score = ratio - 0.005 * abs(extent_cents - delta_hz)
                if score > best_score:
                    best_score = score
                    best_rate = rate
                    best_extent = extent_cents

        ref_seg = slice_for_index(idx, len(track.frame_times), max(8, int(round(config.method_window_s * config.frame_rate_hz))))
        if ref_seg is None:
            continue
        residual_rate, residual_extent = compute_reference_rate_and_extent(track, ref_seg, config)
        agreement = float(clamp01(1.0 - abs(best_rate - residual_rate) / 3.0)) if np.isfinite(residual_rate) else 0.0
        support_score = float(clamp01((best_score - 0.15) / 0.60))
        confidence = float(clamp01(0.40 * support_score + 0.25 * agreement + 0.20 * track.voiced_flag[idx] + 0.15 * clamp01((best_extent - config.min_extent_cents) / 20.0)))
        score = float(clamp01(0.60 * confidence + 0.25 * support_score + 0.15 * agreement))

        features.primary_rate_hz[idx] = best_rate
        features.secondary_rate_hz[idx] = residual_rate
        features.final_rate_hz[idx] = best_rate
        features.final_extent_cents[idx] = best_extent if np.isfinite(best_extent) else residual_extent
        features.confidence[idx] = confidence
        features.window_score[idx] = score
        features.voiced_ratio[idx] = float(track.voiced_flag[idx])
        features.agreement[idx] = agreement
        features.metric_a[idx] = support_score
        features.metric_b[idx] = best_extent
    return features


# ═══════════════════════════════ §5 episode 판정 ═══════════════════════════════
def build_reason_text(method_spec: MethodSpec, episode: Episode) -> str:
    """episode 의 근거 문장(한국어) 생성 — 요약 txt / 그림에 표시."""
    return (
        f"{seconds_to_timestamp(episode.start_time)}-{seconds_to_timestamp(episode.end_time)} 구간은 "
        f"{method_spec.paper_name} 기준으로 {method_spec.primary_rate_label} {episode.primary_rate_hz:.2f}Hz, "
        f"{method_spec.secondary_rate_label} {episode.secondary_rate_hz:.2f}Hz, "
        f"extent {episode.final_extent_cents:.1f} cents, confidence {episode.confidence:.2f}로 파라미터화되었다."
    )


def build_episode_mask(features: WindowFeatures, config: AnalysisConfig) -> np.ndarray:
    """프레임별 특징 → '비브라토 후보' bool 마스크 (episode 판정 1단계).

    조건(모두 만족): detection_band_hz 안의 final_rate, extent ≥ min_extent_cents,
    confidence ≥ 0.28 (코드 고정), window_score ≥ min_score, voiced_ratio ≥ min_voiced_ratio.
    그 뒤 merge_gap_s 보다 짧은 끊김은 이어 붙인다(fill_short_false_gaps).
    """
    rate = np.asarray(features.final_rate_hz, dtype=float)
    extent = np.asarray(features.final_extent_cents, dtype=float)
    mask = (
        np.isfinite(rate)
        & (rate >= config.detection_band_hz[0])
        & (rate <= config.detection_band_hz[1])
        & np.isfinite(extent)
        & (extent >= config.min_extent_cents)
        & np.isfinite(features.confidence)
        & (features.confidence >= 0.28)  # confidence 하한 — AnalysisConfig 에 없는 고정값
        & np.isfinite(features.window_score)
        & (features.window_score >= config.min_score)
        & np.isfinite(features.voiced_ratio)
        & (features.voiced_ratio >= config.min_voiced_ratio)
    )
    gap_frames = max(1, int(round(config.merge_gap_s * config.frame_rate_hz)))
    return fill_short_false_gaps(mask, gap_frames)


def summarize_episode(method_spec: MethodSpec, start: int, end: int, track: PitchTrack, features: WindowFeatures) -> Episode:
    """프레임 구간 [start, end] → Episode (window_score 가중 평균으로 대표값 산출, 점수 0 인 프레임은 가중치 0.1).

    - note_id : 구간 안 최다 노트.  end_time 은 마지막 프레임 시각 + 1 프레임 간격.
    - confidence_tier : ≥0.75 high / ≥0.50 medium / 그 외 low.
    - reason_text : build_reason_text 로 채움.
    """
    seg = slice(start, end + 1)
    weights = np.nan_to_num(features.window_score[seg], nan=0.0)
    weights = np.where(weights > 0, weights, 0.1)
    frame_step = float(np.nanmedian(np.diff(track.frame_times))) if len(track.frame_times) > 1 else 0.0
    episode = Episode(
        note_id=int(np.bincount(track.note_id[seg][track.note_id[seg] >= 0]).argmax()) if np.any(track.note_id[seg] >= 0) else -1,
        start_frame=int(start),
        end_frame=int(end),
        start_time=float(track.frame_times[start]),
        end_time=float(track.frame_times[end] + frame_step),
        duration=float(track.frame_times[end] + frame_step - track.frame_times[start]),
        mean_f0_hz=weighted_nanmean(track.f0_hz[seg], weights),
        center_pitch_hz=weighted_nanmean(track.center_hz[seg], weights),
        center_pitch_cents=weighted_nanmean(track.center_cents[seg], weights),
        primary_rate_hz=weighted_nanmean(features.primary_rate_hz[seg], weights),
        secondary_rate_hz=weighted_nanmean(features.secondary_rate_hz[seg], weights),
        final_rate_hz=weighted_nanmean(features.final_rate_hz[seg], weights),
        final_extent_cents=weighted_nanmean(features.final_extent_cents[seg], weights),
        confidence=weighted_nanmean(features.confidence[seg], weights),
        window_score=weighted_nanmean(features.window_score[seg], weights),
        voiced_ratio=weighted_nanmean(features.voiced_ratio[seg], weights),
        agreement=weighted_nanmean(features.agreement[seg], weights),
        metric_a=weighted_nanmean(features.metric_a[seg], weights),
        metric_b=weighted_nanmean(features.metric_b[seg], weights),
        final_label="existing_vibrato",
        confidence_tier="high",
        reason_text="",
    )
    if episode.confidence < 0.75:
        episode.confidence_tier = "medium"
    if episode.confidence < 0.50:
        episode.confidence_tier = "low"
    episode.reason_text = build_reason_text(method_spec, episode)
    return episode


def merge_episodes(method_spec: MethodSpec, track: PitchTrack, features: WindowFeatures, config: AnalysisConfig) -> List[Episode]:
    """마스크에서 '같은 노트 안 연속 True 구간' 을 찾아 길이 ≥ min_episode_duration_s 인 것만 Episode 로 만든다.

    노트가 바뀌면 마스크가 이어져 있어도 episode 를 끊는다. 반환은 시간순.
    p2b 는 이 Episode 목록의 start_time/end_time/duration 만 구간 정의로 쓰고,
    프레임별 세부 값은 features 를 직접 보간해 쓴다.
    """
    mask = build_episode_mask(features, config)
    frame_step = float(np.nanmedian(np.diff(track.frame_times))) if len(track.frame_times) > 1 else 0.0
    min_frames = max(1, int(round(config.min_episode_duration_s / max(frame_step, EPS))))
    episodes: List[Episode] = []
    idx = 0
    while idx < len(mask):
        if not mask[idx]:
            idx += 1
            continue
        start = idx
        note_id = track.note_id[idx]
        # 마스크가 이어지고 노트도 같은 동안 구간 확장
        while idx + 1 < len(mask) and mask[idx + 1] and track.note_id[idx + 1] == note_id:
            idx += 1
        end = idx
        if end - start + 1 >= min_frames:
            episodes.append(summarize_episode(method_spec, start, end, track, features))
        idx += 1
    return episodes


# ═══════════════ §6 standalone 출력 (CSV / MAT / MATLAB / 그림 / CLI) — 파이프라인(run.py)에서는 사용하지 않음 ═══════════════
def export_table(episodes: Sequence[Episode], output_path: Path):
    """Episode 목록 → CSV. pandas 가 있으면 시간순 정렬한 DataFrame 반환, 없으면 csv 모듈로 기록하고 dict 목록 반환."""
    rows = [asdict(episode) for episode in episodes]
    if pd is not None:
        dataframe = pd.DataFrame(rows)
        if not dataframe.empty:
            dataframe = dataframe.sort_values(["start_time", "end_time"]).reset_index(drop=True)
        dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")
        return dataframe
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Episode.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def summarize_results(method_spec: MethodSpec, episodes: Sequence[Episode], summary_path: Path) -> str:
    """episode 요약 텍스트(영문) 를 만들어 저장하고 반환. tier(high 우선) → confidence 내림차순 → 시간순 정렬."""
    ordered = sorted(episodes, key=lambda item: (item.confidence_tier != "high", -item.confidence, item.start_time))
    lines: List[str] = []
    lines.append(f"{method_spec.paper_name} summary")
    lines.append("")
    lines.append(f"Detected existing vibrato episodes: {len(ordered)}")
    if ordered:
        rates = [item.final_rate_hz for item in ordered if np.isfinite(item.final_rate_hz)]
        extents = [item.final_extent_cents for item in ordered if np.isfinite(item.final_extent_cents)]
        if rates:
            lines.append(f"Rate range: {min(rates):.2f} - {max(rates):.2f} Hz")
        if extents:
            lines.append(f"Extent range: {min(extents):.1f} - {max(extents):.1f} cents")
    lines.append("")
    lines.append("[Detected Existing Vibrato Episodes]")
    if ordered:
        for index, episode in enumerate(ordered, start=1):
            lines.append(
                f"{index}. {seconds_to_timestamp(episode.start_time)}-{seconds_to_timestamp(episode.end_time)} | "
                f"note={episode.note_id} | tier={episode.confidence_tier} | "
                f"rate={episode.final_rate_hz:.2f}Hz | extent={episode.final_extent_cents:.1f}c | "
                f"conf={episode.confidence:.2f}"
            )
            lines.append(f"   {episode.reason_text}")
    else:
        lines.append("No episode passed the detector.")
    text = "\n".join(lines)
    summary_path.write_text(text, encoding="utf-8")
    return text


def save_matlab_data(method_spec: MethodSpec, audio: AudioBundle, track: PitchTrack, features: WindowFeatures, episodes: Sequence[Episode], config: AnalysisConfig, output_path: Path) -> None:
    """audio / track / features / episodes / config 를 .mat 으로 저장 (MATLAB 뷰어용)."""
    payload = {
        "method": {
            "method_id": matlab_cell_column([method_spec.method_id]),
            "paper_name": matlab_cell_column([method_spec.paper_name]),
            "primary_rate_label": matlab_cell_column([method_spec.primary_rate_label]),
            "secondary_rate_label": matlab_cell_column([method_spec.secondary_rate_label]),
            "metric_a_label": matlab_cell_column([method_spec.metric_a_label]),
            "metric_b_label": matlab_cell_column([method_spec.metric_b_label]),
        },
        "audio": {"sr": float(audio.sr), "duration": float(audio.duration), "y_norm": np.asarray(audio.y_norm, dtype=float)},
        "track": {
            "frame_times": np.asarray(track.frame_times, dtype=float),
            "f0_hz": np.asarray(track.f0_hz, dtype=float),
            "center_hz": np.asarray(track.center_hz, dtype=float),
            "residual_cents": np.asarray(track.residual_cents, dtype=float),
            "note_id": np.asarray(track.note_id, dtype=float),
        },
        "features": {
            "primary_rate_hz": np.asarray(features.primary_rate_hz, dtype=float),
            "secondary_rate_hz": np.asarray(features.secondary_rate_hz, dtype=float),
            "final_rate_hz": np.asarray(features.final_rate_hz, dtype=float),
            "final_extent_cents": np.asarray(features.final_extent_cents, dtype=float),
            "confidence": np.asarray(features.confidence, dtype=float),
            "window_score": np.asarray(features.window_score, dtype=float),
            "voiced_ratio": np.asarray(features.voiced_ratio, dtype=float),
            "agreement": np.asarray(features.agreement, dtype=float),
            "metric_a": np.asarray(features.metric_a, dtype=float),
            "metric_b": np.asarray(features.metric_b, dtype=float),
        },
        "episodes": {key: np.asarray([getattr(ep, key) for ep in episodes], dtype=float) for key in [
            "note_id", "start_frame", "end_frame", "start_time", "end_time", "duration",
            "mean_f0_hz", "center_pitch_hz", "center_pitch_cents", "primary_rate_hz",
            "secondary_rate_hz", "final_rate_hz", "final_extent_cents", "confidence",
            "window_score", "voiced_ratio", "agreement", "metric_a", "metric_b"
        ]},
        "episode_text": {
            "final_label": matlab_cell_column([ep.final_label for ep in episodes]),
            "confidence_tier": matlab_cell_column([ep.confidence_tier for ep in episodes]),
            "reason_text": matlab_cell_column([ep.reason_text for ep in episodes]),
        },
        "config": {
            "target_sr": float(config.target_sr),
            "frame_length": float(config.frame_length),
            "hop_length": float(config.hop_length),
            "method_window_s": float(config.method_window_s),
            "search_band_hz": np.asarray(config.search_band_hz, dtype=float),
            "detection_band_hz": np.asarray(config.detection_band_hz, dtype=float),
        },
    }
    savemat(output_path, payload, do_compression=True)


def matlab_path_string(path: Path) -> str:
    """Windows 경로 → MATLAB 문자열 리터럴용 (슬래시 통일, 따옴표 이스케이프)."""
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def find_matlab_executable() -> Optional[str]:
    """PATH 의 matlab, 없으면 고정 경로(D:\\Program\\MATLAB\\R2021b) — 특정 PC 환경 의존. 둘 다 없으면 None."""
    matlab_path = shutil.which("matlab")
    if matlab_path:
        return matlab_path
    fallback = Path(r"D:\Program\MATLAB\R2021b\bin\matlab.exe")
    return str(fallback) if fallback.exists() else None


def write_matlab_viewer_script(method_spec: MethodSpec, mat_path: Path, output_path: Path) -> None:
    """.mat 을 읽어 스펙트로그램+F0, rate 타임라인, 진단 그림을 그리는 MATLAB .m 스크립트를 생성."""
    mat_ref = matlab_path_string(mat_path)
    script = f"""S = load('{mat_ref}');
t = S.track.frame_times(:);
f0 = S.track.f0_hz(:);
center = S.track.center_hz(:);
residual = S.track.residual_cents(:);
primaryRate = S.features.primary_rate_hz(:);
secondaryRate = S.features.secondary_rate_hz(:);
conf = S.features.confidence(:);
score = S.features.window_score(:);
extent = S.features.final_extent_cents(:);
metricA = S.features.metric_a(:);
metricB = S.features.metric_b(:);
audio = S.audio.y_norm(:);
sr = double(S.audio.sr);
frameLength = double(S.config.frame_length);
hopLength = double(S.config.hop_length);
startTimes = S.episodes.start_time(:);
endTimes = S.episodes.end_time(:);

figure('Name', '{method_spec.overview_title}', 'Color', 'w');
[spec, freq, timeSpec] = spectrogram(audio, hamming(frameLength), frameLength-hopLength, frameLength, sr, 'yaxis');
imagesc(timeSpec, freq, mag2db(abs(spec) + eps)); axis xy; colormap turbo; hold on;
for k = 1:numel(startTimes)
    patch([startTimes(k) endTimes(k) endTimes(k) startTimes(k)], [min(freq) min(freq) max(freq) max(freq)], [0 0 0], 'FaceAlpha', 0.18, 'EdgeColor', 'none');
end
plot(t, f0, 'c', 'LineWidth', 1.2);
plot(t, center, 'w--', 'LineWidth', 1.0);
hold off; colorbar; xlabel('Time (s)'); ylabel('Frequency (Hz)'); title('{method_spec.overview_title}');

figure('Name', '{method_spec.paper_name}', 'Color', 'w');
tiledlayout(3,1);
nexttile;
plot(t, f0, 'Color', [0.11 0.21 0.34], 'LineWidth', 1.0); hold on;
plot(t, center, '--', 'Color', [0.16 0.62 0.56], 'LineWidth', 1.0);
yyaxis right; plot(t, residual, 'Color', [0.96 0.64 0.38], 'LineWidth', 0.9); yline(0, ':'); yyaxis left;
grid on; title('Raw F0 / Center / Residual');
nexttile;
plot(t, primaryRate, 'LineWidth', 1.0); hold on; plot(t, secondaryRate, 'LineWidth', 1.0);
yline(double(S.config.detection_band_hz(1)), '--'); yline(double(S.config.detection_band_hz(2)), '--');
legend('{method_spec.primary_rate_label}', '{method_spec.secondary_rate_label}', 'Location', 'best'); grid on; title('Method rates');
nexttile;
plot(t, conf, 'LineWidth', 1.0); hold on; plot(t, score, 'LineWidth', 1.0); plot(t, metricA, 'LineWidth', 0.9); plot(t, metricB, 'LineWidth', 0.9);
legend('confidence', 'window score', '{method_spec.metric_a_label}', '{method_spec.metric_b_label}', 'Location', 'best');
grid on; title('Confidence and method diagnostics');
disp('MATLAB figures loaded.');
"""
    output_path.write_text(script, encoding="utf-8")


def launch_matlab_viewer(script_path: Path) -> bool:
    """MATLAB 을 백그라운드로 띄워 뷰어 스크립트를 실행. 실패/미설치면 False."""
    matlab_exe = find_matlab_executable()
    if not matlab_exe:
        return False
    try:
        subprocess.Popen([matlab_exe, "-nosplash", "-r", f"run('{matlab_path_string(script_path)}');"], cwd=str(script_path.parent))
        return True
    except Exception:
        return False


def plot_spectrogram_overlay(method_spec: MethodSpec, audio: AudioBundle, track: PitchTrack, episodes: Sequence[Episode], config: AnalysisConfig, output_path: Path) -> None:
    """그림 1: 로그 스펙트로그램 + Raw/Center F0 + episode 음영."""
    import librosa.display
    import matplotlib.pyplot as plt

    stft = np.abs(librosa.stft(audio.y_norm, n_fft=config.frame_length, hop_length=config.hop_length))
    stft_db = librosa.amplitude_to_db(np.maximum(stft, EPS), ref=np.max)
    fig, ax = plt.subplots(figsize=(16, 8))
    image = librosa.display.specshow(stft_db, sr=audio.sr, hop_length=config.hop_length, x_axis="time", y_axis="log", cmap="magma", ax=ax)
    for episode in episodes:
        ax.axvspan(episode.start_time, episode.end_time, color="#111111", alpha=0.18)
    ax.plot(track.frame_times, track.f0_hz, color="#69f0ff", linewidth=1.4, label="Raw F0")
    ax.plot(track.frame_times, track.center_hz, color="white", linestyle="--", linewidth=1.0, label="Center F0")
    ax.set_title(method_spec.overview_title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(image, ax=ax, format="%+2.0f dB")
    ax.legend(loc="upper right")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_rate_timeline(method_spec: MethodSpec, track: PitchTrack, features: WindowFeatures, episodes: Sequence[Episode], config: AnalysisConfig, output_path: Path) -> None:
    """그림 2 (3단): F0/center/잔차 · primary/secondary rate + detection band · confidence/score/metric."""
    import matplotlib.pyplot as plt

    time_axis = track.frame_times
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True, constrained_layout=True)
    for axis in axes:
        for episode in episodes:
            axis.axvspan(episode.start_time, episode.end_time, color="#111111", alpha=0.12)
        axis.grid(True, alpha=0.22)

    axes[0].plot(time_axis, track.f0_hz, color="#1d3557", linewidth=1.0, label="Raw F0")
    axes[0].plot(time_axis, track.center_hz, color="#2a9d8f", linestyle="--", linewidth=1.0, label="Center F0")
    axes[0].legend(loc="upper right")
    residual_ax = axes[0].twinx()
    residual_ax.plot(time_axis, track.residual_cents, color="#f4a261", linewidth=0.9)
    residual_ax.axhline(0.0, color="#6c757d", linestyle=":", linewidth=0.8)
    axes[0].set_title("Raw F0 / Center / Residual")

    axes[1].plot(time_axis, features.primary_rate_hz, color="#457b9d", linewidth=1.0, label=method_spec.primary_rate_label)
    axes[1].plot(time_axis, features.secondary_rate_hz, color="#d62828", linewidth=1.0, label=method_spec.secondary_rate_label)
    axes[1].axhspan(config.detection_band_hz[0], config.detection_band_hz[1], color="#bde0fe", alpha=0.25)
    axes[1].legend(loc="upper right")
    axes[1].set_title("Method-specific rate timeline")

    axes[2].plot(time_axis, features.confidence, color="#1d3557", linewidth=1.0, label="Confidence")
    axes[2].plot(time_axis, features.window_score, color="#2a9d8f", linewidth=1.0, label="Window score")
    axes[2].plot(time_axis, features.metric_a, color="#f4a261", linewidth=0.9, label=method_spec.metric_a_label)
    axes[2].plot(time_axis, features.metric_b, color="#6d597a", linewidth=0.9, label=method_spec.metric_b_label)
    axes[2].legend(loc="upper right")
    axes[2].set_title("Method diagnostics")
    axes[2].set_xlabel("Time (s)")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_episode_diagnostics(method_spec: MethodSpec, track: PitchTrack, features: WindowFeatures, episodes: Sequence[Episode], config: AnalysisConfig, output_path: Path) -> None:
    """그림 3: 잔차 전체 + 상위 plot_top_k episode 각각의 F0/center/잔차 확대."""
    import matplotlib.pyplot as plt

    top_episodes = sorted(episodes, key=lambda item: (item.confidence_tier != "high", -item.confidence, item.start_time))[: max(1, config.plot_top_k)]
    fig, axes = plt.subplots(1 + len(top_episodes), 1, figsize=(16, 4 + 2.8 * (1 + len(top_episodes))), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    axes[0].plot(track.frame_times, track.residual_cents, color="#f4a261", linewidth=0.9)
    axes[0].axhline(0.0, color="#6c757d", linestyle=":", linewidth=0.8)
    axes[0].set_title(f"{method_spec.paper_name} residual overview")
    axes[0].grid(True, alpha=0.22)
    for episode in top_episodes:
        axes[0].axvspan(episode.start_time, episode.end_time, color="#111111", alpha=0.15)
    for axis, episode in zip(axes[1:], top_episodes):
        seg = slice(episode.start_frame, episode.end_frame + 1)
        axis.plot(track.frame_times[seg], track.f0_hz[seg], color="#1d3557", linewidth=1.0, label="Raw F0")
        axis.plot(track.frame_times[seg], track.center_hz[seg], color="#2a9d8f", linestyle="--", linewidth=1.0, label="Center F0")
        residual_axis = axis.twinx()
        residual_axis.plot(track.frame_times[seg], track.residual_cents[seg], color="#f4a261", linewidth=0.9)
        axis.set_title(
            f"{seconds_to_timestamp(episode.start_time)}-{seconds_to_timestamp(episode.end_time)} | "
            f"{method_spec.primary_rate_label} {episode.primary_rate_hz:.2f}Hz | conf {episode.confidence:.2f}"
        )
        axis.grid(True, alpha=0.22)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_top_episode_summary(method_spec: MethodSpec, episodes: Sequence[Episode], config: AnalysisConfig, output_path: Path) -> None:
    """그림 4: 상위 episode 표를 텍스트로 그린 이미지."""
    import matplotlib.pyplot as plt

    top_episodes = sorted(episodes, key=lambda item: (item.confidence_tier != "high", -item.confidence, item.start_time))[: max(1, config.plot_top_k)]
    fig, ax = plt.subplots(figsize=(18, 1.4 * max(5, len(top_episodes) + 2)))
    ax.axis("off")
    ax.set_title(method_spec.paper_name, loc="left", fontsize=14, pad=12)
    ax.text(0.02, 0.94, "time range | note | final rate | extent | conf | agreement | tier", fontsize=11, family="monospace", fontweight="bold")
    y = 0.86
    step = 0.12 if len(top_episodes) <= 5 else 0.09
    for episode in top_episodes:
        color = "#2a9d8f" if episode.confidence_tier == "high" else "#f4a261" if episode.confidence_tier == "medium" else "#d62828"
        line = (
            f"{seconds_to_timestamp(episode.start_time)}-{seconds_to_timestamp(episode.end_time)} | "
            f"N{episode.note_id:03d} | {episode.final_rate_hz:5.2f} Hz | {episode.final_extent_cents:6.1f} c | "
            f"{episode.confidence:4.2f} | {episode.agreement:4.2f} | {episode.confidence_tier}"
        )
        ax.text(0.02, y, line, fontsize=10, family="monospace", color=color)
        ax.text(0.02, y - step * 0.40, episode.reason_text, fontsize=8.4, color="#444444")
        y -= step
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_parser(method_spec: MethodSpec) -> argparse.ArgumentParser:
    """standalone CLI 인자 정의.

    기본값은 AnalysisConfig 기본값과 같아야 한다 — 현재는 수동으로 맞춰 둔 중복이라
    한쪽만 고치면 CLI 와 파이프라인의 동작이 어긋난다 (모듈화 점검 항목).
    """
    parser = argparse.ArgumentParser(description=f"{method_spec.paper_name} standalone detector")
    parser.add_argument("audio_path", nargs="?", default="Pavarotti_vocals.wav")
    parser.add_argument("--output-dir", default=f"outputs_{method_spec.output_prefix}")
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--frame-length", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=220)
    parser.add_argument("--fmin", type=float, default=float(librosa.note_to_hz("C2")))
    parser.add_argument("--fmax", type=float, default=float(librosa.note_to_hz("C7")))
    parser.add_argument("--flat-window", type=float, default=0.45)
    parser.add_argument("--method-window", type=float, default=0.30)
    parser.add_argument("--note-jump-cents", type=float, default=90.0)
    parser.add_argument("--search-band-min", type=float, default=3.0)
    parser.add_argument("--search-band-max", type=float, default=11.0)
    parser.add_argument("--detect-band-min", type=float, default=4.0)
    parser.add_argument("--detect-band-max", type=float, default=8.0)
    parser.add_argument("--min-voiced-ratio", type=float, default=0.80)
    parser.add_argument("--min-extent", type=float, default=10.0)
    parser.add_argument("--min-score", type=float, default=0.45)
    parser.add_argument("--min-episode-duration", type=float, default=1.50)
    parser.add_argument("--merge-gap", type=float, default=0.20)
    parser.add_argument("--plot-top-k", type=int, default=6)
    parser.add_argument("--use-harmonic-pitch", action="store_true")
    parser.add_argument("--use-original-pitch", action="store_true", help=argparse.SUPPRESS)
    return parser


def main_for_method(method_id: str) -> None:
    """standalone 실행 진입점: 오디오 → 8단계(로드/F0/특징/episode/그림/CSV/MAT/요약+MATLAB) → outputs_<방식>/ 에 저장.

    파이프라인(run.py → p2 → p2b)은 이 함수를 거치지 않고 §3~§5 함수들을 직접 호출한다.
    """
    method_spec = METHOD_SPECS[method_id]
    parser = build_parser(method_spec)
    args = parser.parse_args()
    config = AnalysisConfig(
        target_sr=args.sr,
        use_harmonic_for_pitch=bool(args.use_harmonic_pitch and not args.use_original_pitch),
        frame_length=args.frame_length,
        hop_length=args.hop_length,
        fmin=args.fmin,
        fmax=args.fmax,
        flat_window_s=args.flat_window,
        note_jump_cents=args.note_jump_cents,
        method_window_s=args.method_window,
        search_band_hz=(args.search_band_min, args.search_band_max),
        detection_band_hz=(args.detect_band_min, args.detect_band_max),
        min_voiced_ratio=args.min_voiced_ratio,
        min_extent_cents=args.min_extent,
        min_score=args.min_score,
        min_episode_duration_s=args.min_episode_duration,
        merge_gap_s=args.merge_gap,
        plot_top_k=args.plot_top_k,
    )
    audio_path = resolve_audio_path(args.audio_path)
    output_dir = create_run_output_dir(Path(args.output_dir), audio_path)
    total_steps = 8
    start_clock = time.perf_counter()
    last_stage_clock = start_clock

    def stage_done(step: int, label: str) -> None:
        nonlocal last_stage_clock
        now = time.perf_counter()
        stage_elapsed = now - last_stage_clock
        total_elapsed = now - start_clock
        log_progress(
            f"[{step}/{total_steps}] {label} 완료 | "
            f"step={format_elapsed(stage_elapsed)} | total={format_elapsed(total_elapsed)}"
        )
        last_stage_clock = now

    log_progress(f"{method_spec.paper_name} analysis start: {audio_path}")
    log_progress(
        "Config | "
        f"sr={config.target_sr}, frame={config.frame_length}, hop={config.hop_length}, "
        f"flat_window={config.flat_window_s:.2f}s, method_window={config.method_window_s:.2f}s, "
        f"detect_band={config.detection_band_hz[0]:.1f}-{config.detection_band_hz[1]:.1f}Hz"
    )

    log_progress(f"[1/{total_steps}] audio loading + preprocessing")
    audio = load_and_preprocess_audio(audio_path, config)
    stage_done(1, "audio loading + preprocessing")

    log_progress(f"[2/{total_steps}] F0 extraction + center/residual estimation")
    track = extract_pitch_track(audio, config)
    voiced_frames = int(np.sum(track.voiced_flag))
    total_frames = int(len(track.frame_times))
    log_progress(
        f"Pitch track debug | frames={total_frames}, voiced_frames={voiced_frames}, "
        f"duration={audio.duration:.2f}s"
    )
    stage_done(2, "F0 extraction + center/residual estimation")

    compute_fn: Dict[str, Callable[[AudioBundle, PitchTrack, AnalysisConfig], WindowFeatures]] = {
        "spectrum_modelling": compute_features_spectrum_modelling,
        "spectral_envelopes_distortion": compute_features_spectral_envelopes_distortion,
        "analytic_signal": compute_features_analytic_signal,
        "minima_maxima_detection": compute_features_minima_maxima_detection,
    }
    log_progress(f"[3/{total_steps}] feature computation ({method_id})")
    features = compute_fn[method_id](audio, track, config)
    finite_rate = int(np.sum(np.isfinite(features.final_rate_hz)))
    finite_extent = int(np.sum(np.isfinite(features.final_extent_cents)))
    candidate_like = int(
        np.sum(
            np.isfinite(features.final_rate_hz)
            & np.isfinite(features.final_extent_cents)
            & np.isfinite(features.window_score)
            & (features.window_score >= config.min_score)
        )
    )
    log_progress(
        f"Feature debug | finite_rate={finite_rate}, finite_extent={finite_extent}, "
        f"score_pass_frames={candidate_like}"
    )
    stage_done(3, "feature computation")

    log_progress(f"[4/{total_steps}] episode merging")
    episodes = merge_episodes(method_spec, track, features, config)
    total_detected_duration = float(np.nansum([ep.duration for ep in episodes])) if episodes else 0.0
    log_progress(
        f"Episode debug | detected={len(episodes)}, detected_duration={total_detected_duration:.2f}s"
    )
    stage_done(4, "episode merging")

    prefix = method_spec.output_prefix
    overview_path = output_dir / f"{prefix}_vibrato_spectrogram_overlay.png"
    timeline_path = output_dir / f"{prefix}_vibrato_rate_timeline.png"
    diagnostics_path = output_dir / f"{prefix}_vibrato_episode_diagnostics.png"
    summary_fig_path = output_dir / f"{prefix}_vibrato_top_segments.png"
    csv_path = output_dir / "detected_existing_vibrato_segments.csv"
    mat_path = output_dir / f"{prefix}_vibrato_analysis.mat"
    matlab_script_path = output_dir / f"open_{prefix}_vibrato_figures.m"
    summary_path = output_dir / "top_vibrato_summary.txt"

    pre_plot_elapsed = time.perf_counter() - start_clock
    log_progress(f"[DEBUG] plot 시작 직전 누적 시간 = {format_elapsed(pre_plot_elapsed)}")

    log_progress(f"[5/{total_steps}] plot generation")
    plot_spectrogram_overlay(method_spec, audio, track, episodes, config, overview_path)
    plot_rate_timeline(method_spec, track, features, episodes, config, timeline_path)
    plot_episode_diagnostics(method_spec, track, features, episodes, config, diagnostics_path)
    plot_top_episode_summary(method_spec, episodes, config, summary_fig_path)
    stage_done(5, "plot generation")

    log_progress(f"[6/{total_steps}] table export")
    table = export_table(episodes, csv_path)
    stage_done(6, "table export")

    log_progress(f"[7/{total_steps}] mat export + matlab viewer script")
    save_matlab_data(method_spec, audio, track, features, episodes, config, mat_path)
    write_matlab_viewer_script(method_spec, mat_path, matlab_script_path)
    stage_done(7, "mat export + matlab viewer script")

    log_progress(f"[8/{total_steps}] summary text + matlab launch")
    summary_text = summarize_results(method_spec, episodes, summary_path)
    matlab_started = launch_matlab_viewer(matlab_script_path)
    stage_done(8, "summary text + matlab launch")

    log_progress(f"[DONE] total analysis time = {format_elapsed(time.perf_counter() - start_clock)}")

    print(summary_text)
    print("")
    print("[Saved Files]")
    for path in [overview_path, timeline_path, diagnostics_path, summary_fig_path, csv_path, mat_path, matlab_script_path, summary_path]:
        print(f"- {path}")
    print("")
    print("[MATLAB]")
    print(f"- auto_launch={'yes' if matlab_started else 'no'}")
    if pd is not None and hasattr(table, "empty") and not table.empty:
        print("")
        print("[DataFrame Preview]")
        print(table.head(min(len(table), 10)).to_string(index=False))


# 단독 실행 시 analytic_signal 방식으로 분석 (다른 방식은 method_id 만 바꿔 호출)
if __name__ == "__main__":
    main_for_method("analytic_signal")
