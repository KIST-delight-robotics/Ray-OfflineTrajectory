"""[2단계-b] 비브라토 검출 결과 → 입 모터 떨림(delta) 변환.

역할
  p2a_vibrato_detector 가 보컬 wav 에서 찾아낸 '비브라토 episode(구간)' 와
  프레임별 특징(속도 Hz·폭 cents·신뢰도)을 받아, 입 모터 궤적(base_mouth, tick 단위)에
  사인파 떨림(delta)을 더해 비브라토가 과장된 입 궤적(corrected_mouth)을 만든다.

데이터 흐름
  보컬 wav ──p2a──▶ VibratoDetectionBundle(episodes + 프레임별 rate/extent/confidence)
                        │
  base_mouth(tick) ─────┴──apply_vibrato_exaggeration(cfg)──▶ VibratoMotionBundle
                                                              ├ corrected_mouth  ← Pi 로 가는 mouth col0
                                                              ├ roll_comp_mouth  ← mouth col2 (roll 보상용)
                                                              └ delta/active/rate/extent/conf (디버그 csv)

사용처
  p2_offline_motion.offline_motion_generation() 이 detect_vibrato_episodes() →
  apply_vibrato_exaggeration() 순으로 호출한다. 튜닝 손잡이는 VibratoMotionConfig 참고.

역할 분담 (검출 vs 표현)
  p2a.AnalysisConfig      : '어디가 비브라토 구간인가' (검출 파라미터)
  p2b.VibratoMotionConfig : '검출된 구간을 모터로 어떻게 떨게 할까' (표현 파라미터)

단위 규약
  tick  : 입 모터 위치. p2 에서는 0 = 다문 상태(max_MOUTH), 250 = 최대 벌림(min_MOUTH) 으로 넘긴다.
          (p2 의 변수명 max/min 은 '음성 에너지' 기준이라 tick 크기와 반대이므로 주의)
  s     : 초.  Hz : 초당 떨림 횟수.  cents : 음정 흔들림 폭(100 cents = 반음).
  시각축: 입 궤적은 40 ms/행(25 fps), p2a 프레임은 ≈10 ms(≈100 fps) → 보간해서 맞춘다.

원본: offline_pipeline/s3b (VibratoMotionConfig). 로직은 그대로이고 주석만 보강되었다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

import p2a_vibrato_detector as analytic


EPS = 1e-8  # 0 나눗셈 방지용 아주 작은 수


@dataclass
class VibratoMotionConfig:
    """비브라토 떨림을 입 모션 궤적에 더할 때 쓰는 조절 변수 모음.

    아래 (1)~(7) 그룹은 '비브라토'를 만들 때 건드릴 수 있는 측면별로 묶여 있다.
    각 변수는 apply_vibrato_exaggeration() 에서 실제로 사용된다(본문 주석 ①~⑧ 참조).

    파이프라인(run.py)에서 CLI 로 노출된 손잡이는 3개뿐이다:
      --vibrato-max-ticks          → vibrato_max_ticks
      --vibrato-extent-reference   → vibrato_extent_reference_cents
      --vibrato-fade-in-s          → vibrato_fade_in_s
    mouth_closed_tick / mouth_open_tick 은 p2 가 자기 상수(max_MOUTH=0, min_MOUTH=250)로 채운다.
    나머지는 이 파일의 기본값을 직접 고쳐서 조절한다.

    (3) 그룹의 게이트는 p2a 가 이미 episode 로 판정한 구간 안에서 프레임 단위로
    한 번 더 좁히는 2차 필터다 (p2a: rate 4~8 Hz, extent ≥10 c → p2b: 4.4~7.6 Hz, ≥12 c).
    """

    # ── (1) 입 모터 좌표계 : 떨림이 더해질 기준 범위 (tick 단위) ─────────────────────────
    mouth_closed_tick: float = 0.0        # [tick] 입을 다물었을 때의 모터 위치 (p2: max_MOUTH=0)
    mouth_open_tick: float = 250.0        # [tick] 입을 최대로 벌렸을 때의 모터 위치 (p2: min_MOUTH=250)
    # [tick | None] 최대 벌림을 넘어서까지 떨림을 허용할 여유.
    #   None 이면 vibrato_max_ticks 와 같은 값을 쓴다 (손잡이 하나로 통일).
    vibrato_overflow_ticks: float | None = None

    # ── (2) 떨림 진폭 : 입을 얼마나 크게 떨게 할지 ──────────────────────────────────────
    vibrato_max_ticks: float = 18.0                # [tick]  떨림 최대 진폭 (extent 가 reference 일 때 100%)
    vibrato_extent_reference_cents: float = 45.0   # [cents] 이 음정 폭을 진폭 100% 기준으로 정규화 (최대 160% 까지 허용)

    # ── (3) 비브라토 인정 게이트 : 어떤 프레임을 '진짜 비브라토'로 볼지 ─────────────────────
    vibrato_min_rate_hz: float = 4.4         # [Hz]    인정할 떨림 속도 하한 (p2a detection_band 4.0 보다 좁음)
    vibrato_max_rate_hz: float = 7.6         # [Hz]    인정할 떨림 속도 상한 (p2a detection_band 8.0 보다 좁음)
    vibrato_min_extent_cents: float = 12.0   # [cents] 인정할 최소 음정 폭 (p2a min_extent 10 보다 엄격)

    # ── (4) 구간 다듬기 : 게이트 통과 마스크를 시간축에서 정리 ────────────────────────────
    vibrato_min_apply_duration_s: float = 0.8   # [s] episode 전체 길이가 이보다 짧으면 통째로 무시
    vibrato_min_active_run_s: float = 0.55      # [s] 게이트 통과 구간이 이보다 짧게 이어지면 제거
    vibrato_merge_gap_s: float = 0.12           # [s] 게이트 통과 구간 사이 이보다 짧은 끊김은 이어 붙임

    # ── (5) 부드러움 : 떨림이 튀거나 뚝 끊기지 않게 ────────────────────────────────────
    vibrato_fade_in_s: float = 0.08       # [s] episode 시작 페이드인 (사인 1/4주기 곡선)
    vibrato_fade_out_s: float = 0.08      # [s] episode 끝 페이드아웃
    amplitude_smoothing_s: float = 0.14   # [s] 진폭 곡선 가우시안 스무딩 반경 (0 이면 끔)

    # ── (6) 입 벌림 정도에 따른 진폭 변조 : 크게 벌린 음에서 더 극적으로 ────────────────────
    base_mouth_min_amplitude_scale: float = 0.5    # [배율] 입이 거의 닫혔을 때의 진폭 배율 하한
    base_mouth_amplitude_power: float = 1.2        # [지수] 벌림비율^power 로 배율 곡선 모양 결정 (>1: 크게 벌릴 때만 커짐)
    dramatic_mouth_threshold_ratio: float = 0.82   # [0~1] 이 벌림 비율 이상을 '드라마틱' 구간으로 간주
    dramatic_mouth_boost_scale: float = 1.15       # [배율] 드라마틱 구간 진폭 부스트 (최종 배율은 1.2 로 클립)

    # ── (7) roll 보상 출력 : roll 축 보상용 mouth 열(col2)에 섞을 비브라토 비율 ────────────
    roll_comp_vibrato_scale: float = 0.2   # [배율] 0 = roll 보상 열은 떨림 없음, 1 = mouth 와 동일하게 떨림

    @property
    def resolved_overflow_ticks(self) -> float:
        """[tick] 실제 적용할 overflow 여유 — None 이면 vibrato_max_ticks 로 대체."""
        if self.vibrato_overflow_ticks is None:
            return float(self.vibrato_max_ticks)
        return float(self.vibrato_overflow_ticks)

    @property
    def resolved_mouth_open_tick(self) -> float:
        """[tick] 최대 벌림 위치 (현재는 mouth_open_tick 을 float 로 돌려주는 단순 통과)."""
        return float(self.mouth_open_tick)


@dataclass
class VibratoDetectionBundle:
    """p2a 검출 결과 묶음 (detect_vibrato_episodes 의 반환값).

    프레임 시각축은 track.frame_times (p2a hop 기준, 기본 22050/220 ≈ 100 fps) 이고
    입 궤적 시각축(40 ms = 25 fps)과 다르므로, 사용할 때 _interp_track_values 로 맞춘다.
    """

    analysis_config: analytic.AnalysisConfig   # 검출에 사용한 p2a 설정 (기록·재현용)
    track: analytic.PitchTrack                 # 프레임별 F0·중심 음정·잔차·노트 id
    features: analytic.WindowFeatures          # 프레임별 rate[Hz]·extent[cents]·confidence 등 (미계산 프레임 NaN)
    episodes: Sequence[analytic.Episode]       # 최종 비브라토 구간 목록 (start_time/end_time/duration 사용)


@dataclass
class VibratoMotionBundle:
    """apply_vibrato_exaggeration 의 반환값. 모든 배열은 base_mouth 와 길이가 같다(40 ms/행).

    Pi 로 나가는 것은 corrected_mouth(col0), roll_comp_mouth(col2) 두 개이고
    나머지는 <곡>-delta-big_debug.csv 로 저장되는 진단용이다.
    """

    corrected_mouth: np.ndarray        # [tick]  base_mouth + 떨림 delta (모터 범위로 클립 완료) ← 최종 입 궤적
    roll_comp_mouth: np.ndarray        # [tick]  base_mouth + delta × roll_comp_vibrato_scale ← roll 보상용
    vibrato_delta: np.ndarray          # [tick]  더해진 떨림 성분만 (양·음 사인파, episode 밖 0)
    vibrato_active: np.ndarray         # [0/1]   해당 프레임이 게이트를 통과해 실제로 떨렸는지
    vibrato_rate_hz: np.ndarray        # [Hz]    episode 안 프레임의 떨림 속도 (밖은 0)
    vibrato_extent_cents: np.ndarray   # [cents] episode 안 프레임의 음정 폭 (밖은 0)
    vibrato_confidence: np.ndarray     # [0~1]   episode 안 프레임의 검출 신뢰도 (밖은 0)


def detect_vibrato_episodes(audio_path) -> VibratoDetectionBundle:
    """보컬 wav → p2a 검출 파이프라인 실행 → VibratoDetectionBundle.

    p2a 의 4가지 검출 방식 중 'analytic_signal'(힐베르트 변환 기반)만 사용하며,
    AnalysisConfig 는 p2a 기본값을 그대로 쓴다(원본 vibrato_detector.py 와 동일 설정).

    단계
      1) load_and_preprocess_audio        : wav 로드(22050 Hz mono) + 피크 정규화
      2) extract_pitch_track              : pyin F0(디스크 캐시) → 중심 음정/잔차 → 노트 분할
      3) compute_features_analytic_signal : 잔차 대역통과 → 힐베르트 → 프레임별 rate/extent/conf
      4) merge_episodes                   : 프레임 마스크 → 노트 안 연속 구간 → Episode 목록

    비용 : pyin 이 대부분(곡당 수십 초~수 분). data/pyin_cache/ 에 캐시되어 2회차부터 수 초.
    """
    # 원본 vibrato_detector.py 에 정의된 검출 설정을 그대로 재사용한다.
    analysis_cfg = analytic.AnalysisConfig()
    audio = analytic.load_and_preprocess_audio(audio_path, analysis_cfg)              # 1) 로드 + 정규화
    track = analytic.extract_pitch_track(audio, analysis_cfg)                          # 2) F0 → center/residual → note
    features = analytic.compute_features_analytic_signal(audio, track, analysis_cfg)  # 3) 프레임별 rate/extent/conf
    episodes = analytic.merge_episodes(analytic.METHOD_SPECS["analytic_signal"], track, features, analysis_cfg)  # 4) episode 목록
    return VibratoDetectionBundle(
        analysis_config=analysis_cfg,
        track=track,
        features=features,
        episodes=episodes,
    )


def _build_episode_envelope(rel_t: np.ndarray, duration: float, cfg: VibratoMotionConfig) -> np.ndarray:
    """episode 시작/끝을 부드럽게 만드는 0~1 포락선.

    rel_t    : [s] episode 시작 기준 상대 시각 (길이 M)
    duration : [s] episode 전체 길이
    반환     : 페이드인 × 페이드아웃 곡선 (0~1, 길이 M). 각 페이드는 직선 대신 sin(π/2·x) 로 더 부드럽게.
    """
    if cfg.vibrato_fade_in_s <= 0.0 and cfg.vibrato_fade_out_s <= 0.0:
        return np.ones_like(rel_t, dtype=float)
    fade_in = np.clip(rel_t / max(cfg.vibrato_fade_in_s, EPS), 0.0, 1.0)        # 0→1 (시작 후 fade_in_s 동안)
    release_t = np.maximum(0.0, duration - rel_t)                                 # [s] 끝까지 남은 시간
    fade_out = np.clip(release_t / max(cfg.vibrato_fade_out_s, EPS), 0.0, 1.0)   # 1→0 (끝나기 fade_out_s 전부터)
    return np.sin(0.5 * np.pi * fade_in) * np.sin(0.5 * np.pi * fade_out)


def _interp_track_values(
    source_times: np.ndarray,
    source_values: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    """p2a 프레임 시각축(≈10 ms) 의 값을 입 궤적 시각축(40 ms)으로 선형 보간.

    source_times / source_values : p2a track.frame_times 와 그에 대응하는 프레임별 값 (NaN 허용)
    target_times                 : [s] 값을 알고 싶은 시각들 (입 궤적 행 시각)
    NaN(특징 미계산 프레임)은 0 으로 바꾼 뒤 보간하므로, rate/extent 하한 게이트에서 자연히 걸러진다.
    범위 밖은 양 끝 값으로 고정(left/right).
    """
    clean = np.nan_to_num(source_values.astype(float), nan=0.0)
    if source_times.size == 0:
        return np.zeros_like(target_times, dtype=float)
    return np.interp(target_times, source_times, clean, left=clean[0], right=clean[-1])


def _smooth_track(values: np.ndarray, times_s: np.ndarray, smoothing_s: float) -> np.ndarray:
    """진폭 곡선을 가우시안 커널로 스무딩 (진폭이 프레임 단위로 뚝뚝 바뀌는 것을 완화).

    values      : 스무딩할 배열 (길이 M)
    times_s     : [s] 각 값의 시각 — 샘플 간격(dt) 추정에만 사용
    smoothing_s : [s] 커널 반경. radius = smoothing_s/dt 프레임, sigma = radius/2.5
    ※ np.convolve(mode="same") 은 양 끝을 0 으로 패딩하므로 가장자리 진폭이 조금 줄어든다
      (페이드인/아웃과 방향이 같아 문제되지 않음).
    """
    if values.size <= 2 or smoothing_s <= 0.0 or times_s.size <= 1:
        return values

    dt = float(np.median(np.diff(times_s)))   # [s] 대표 샘플 간격 (보통 0.04)
    if dt <= 0.0:
        return values

    radius = max(1, int(round(smoothing_s / dt)))   # 커널 반경(샘플 수)
    if radius <= 1:
        return values

    x = np.arange(-radius, radius + 1, dtype=float)
    sigma = max(radius / 2.5, EPS)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= np.sum(kernel)                  # 합이 1 인 정규화 가우시안
    return np.convolve(values, kernel, mode="same")


def _merge_short_gaps(mask: np.ndarray, times_s: np.ndarray, max_gap_s: float) -> np.ndarray:
    """bool 마스크에서 True 사이에 끼인 짧은 False 구간(길이 ≤ max_gap_s)을 True 로 메운다.

    양 끝(마스크 시작 전/끝 후)의 False 는 '끼인' 것이 아니므로 그대로 둔다.
    p2a.fill_short_false_gaps 와 같은 역할이지만 프레임 수 대신 시간[s] 기준.
    """
    if mask.size == 0:
        return mask

    out = mask.astype(bool).copy()
    n = out.size
    i = 0
    while i < n:
        if out[i]:
            i += 1
            continue
        j = i
        while j < n and not out[j]:   # [i, j) = 연속 False 구간
            j += 1
        if i > 0 and j < n:           # 앞뒤가 모두 True 인 '끼인' 구간만 대상
            gap_dur = times_s[j - 1] - times_s[i]
            if gap_dur <= max_gap_s:
                out[i:j] = True
        i = j
    return out


def _remove_short_runs(mask: np.ndarray, times_s: np.ndarray, min_run_s: float) -> np.ndarray:
    """bool 마스크에서 길이 < min_run_s 인 짧은 True 조각을 False 로 지운다 (튀는 오검출 억제)."""
    if mask.size == 0:
        return mask

    out = mask.astype(bool).copy()
    n = out.size
    i = 0
    while i < n:
        if not out[i]:
            i += 1
            continue
        j = i
        while j < n and out[j]:       # [i, j) = 연속 True 구간
            j += 1
        run_dur = times_s[j - 1] - times_s[i]
        if run_dur < min_run_s:
            out[i:j] = False
        i = j
    return out


def apply_vibrato_exaggeration(
    times_s: np.ndarray,
    base_mouth: np.ndarray,
    detection: VibratoDetectionBundle,
    cfg: VibratoMotionConfig,
) -> VibratoMotionBundle:
    """검출된 비브라토 episode 마다 입 궤적에 사인파 떨림을 더한다.

    인자
      times_s    : [s]    입 궤적 각 행의 시각 (40 ms 간격, 길이 N)
      base_mouth : [tick] 비브라토 적용 전 입 궤적 (길이 N, closed=0 … open=250)
      detection  : p2a 검출 결과 (episodes + 프레임별 rate/extent/conf)
      cfg        : 표현 설정 (VibratoMotionConfig)

    episode 하나당 처리 순서 (본문 주석 ①~⑧ 과 대응)
      ① episode 길이 게이트
      ② p2a 프레임 특징(rate/extent/conf)을 입 궤적 시각축으로 보간
      ③ rate/extent 2차 게이트 + 짧은 끊김 메우기/짧은 조각 제거 → active 마스크
      ④ extent → 진폭[tick] 변환 (extent_reference 에서 100%)
      ⑤ 입 벌림 정도에 따른 진폭 변조 (+드라마틱 부스트)
      ⑥ 페이드 포락선 → 모터 범위 '여유 공간' 클립 → 진폭 스무딩
      ⑦ rate 를 시간 적분해 위상 → 사인파 delta
      ⑧ 결과 누적
    마지막에 base+delta 를 모터 범위로 클립해 corrected_mouth / roll_comp_mouth 를 만든다.
    """
    episodes = detection.episodes

    # 입 벌림 비율 (0 = 다문, 1 = 최대 벌림) — ⑤ 진폭 변조에 사용. 전체 길이에 대해 한 번만 계산.
    global_open_ratio = np.clip(
        (base_mouth - cfg.mouth_closed_tick)
        / max(cfg.resolved_mouth_open_tick - cfg.mouth_closed_tick, EPS),
        0.0,
        1.0,
    )
    # 출력 배열 초기화 (episode 밖은 전부 0 으로 남음)
    delta = np.zeros_like(base_mouth, dtype=float)          # [tick]  더할 떨림
    active = np.zeros_like(base_mouth, dtype=float)         # [0/1]   실제 떨린 프레임
    rate_track = np.zeros_like(base_mouth, dtype=float)     # [Hz]    디버그용
    extent_track = np.zeros_like(base_mouth, dtype=float)   # [cents] 디버그용
    conf_track = np.zeros_like(base_mouth, dtype=float)     # [0~1]   디버그용

    for episode in episodes:
        # ① episode 길이 게이트 — 너무 짧은 구간은 모터가 표현하기 전에 끝나므로 건너뜀.
        #    (p2a 기본 min_episode_duration_s=1.5 > 0.8 이라 기본 설정에서는 항상 통과)
        if episode.duration < cfg.vibrato_min_apply_duration_s:
            continue

        # episode 시간 범위 [start, end] 에 들어오는 입 궤적 행 선택
        seg = (times_s >= episode.start_time) & (times_s <= episode.end_time)
        if not np.any(seg):
            continue

        rel_t = times_s[seg] - episode.start_time   # [s] episode 시작 기준 상대 시각 (페이드용)
        local_times = times_s[seg]                  # [s] 절대 시각 (보간·dt 계산용)
        # ② p2a 프레임별 특징(≈10 ms 간격)을 입 궤적 시각축(40 ms)으로 보간
        local_extent = _interp_track_values(
            detection.track.frame_times,
            detection.features.final_extent_cents,
            local_times,
        )
        local_conf = _interp_track_values(
            detection.track.frame_times,
            detection.features.confidence,
            local_times,
        )
        local_rate_track = _interp_track_values(
            detection.track.frame_times,
            detection.features.final_rate_hz,
            local_times,
        )

        # ③ 프레임별 2차 게이트: 속도 범위 & 최소 폭 → 짧은 끊김 메우기 → 짧은 조각 제거
        active_mask = (
            (local_rate_track >= cfg.vibrato_min_rate_hz)
            & (local_rate_track <= cfg.vibrato_max_rate_hz)
            & (local_extent >= cfg.vibrato_min_extent_cents)
        )
        active_mask = _merge_short_gaps(active_mask, local_times, cfg.vibrato_merge_gap_s)
        active_mask = _remove_short_runs(active_mask, local_times, cfg.vibrato_min_active_run_s)
        activity_weight = active_mask.astype(float)   # 0/1 — 게이트 밖 프레임은 진폭 0

        episode_envelope = _build_episode_envelope(rel_t, episode.duration, cfg)   # ⑥ 에서 곱할 페이드 포락선

        # ④ 직접 매핑: 음정 폭(extent) → 진폭[tick], 떨림 속도(rate) → 사인파 주파수.
        #    extent_reference 에서 100%, 그 이상은 160% 까지만 허용.
        extent_curve = np.clip(local_extent / max(cfg.vibrato_extent_reference_cents, EPS), 0.0, 1.6)
        amp_ticks = cfg.vibrato_max_ticks * extent_curve
        amp_ticks = np.where(activity_weight > 0.0, amp_ticks, 0.0)
        # ⑤ 입 벌림 정도에 따른 진폭 변조:
        #    scale = min_scale + (1 − min_scale)·open_ratio^power, 드라마틱 구간은 ×boost, 최종 [min_scale, 1.2] 로 클립
        base_seg = base_mouth[seg]
        open_ratio_seg = global_open_ratio[seg]
        mouth_amp_scale = cfg.base_mouth_min_amplitude_scale + (
            1.0 - cfg.base_mouth_min_amplitude_scale
        ) * np.power(open_ratio_seg, cfg.base_mouth_amplitude_power)
        dramatic_mask = open_ratio_seg >= cfg.dramatic_mouth_threshold_ratio
        mouth_amp_scale = np.where(
            dramatic_mask,
            mouth_amp_scale * cfg.dramatic_mouth_boost_scale,
            mouth_amp_scale,
        )
        mouth_amp_scale = np.clip(mouth_amp_scale, cfg.base_mouth_min_amplitude_scale, 1.2)
        amp_ticks = amp_ticks * mouth_amp_scale
        # ⑥ 페이드 적용 → 모터 범위를 넘지 않도록 '여유 공간'으로 진폭 제한 → 진폭 곡선 스무딩
        amp_ticks = amp_ticks * episode_envelope
        room_to_close = np.maximum(0.0, base_seg - cfg.mouth_closed_tick)   # [tick] 닫히는 쪽 여유
        room_to_open = np.maximum(
            0.0,
            cfg.resolved_mouth_open_tick + cfg.resolved_overflow_ticks - base_seg,   # [tick] 벌리는 쪽 여유 (+overflow)
        )
        local_amp = np.minimum(amp_ticks, np.minimum(room_to_close, room_to_open))
        if cfg.amplitude_smoothing_s > 0.0:
            local_amp = _smooth_track(local_amp, local_times, cfg.amplitude_smoothing_s)
        # ⑦ 위상 = 2π·∫rate dt — 프레임마다 rate 가 달라도 위상이 끊기지 않게 누적 적분 → 사인파 delta
        motor_rate_hz = local_rate_track
        if local_times.size > 1:
            dt = np.diff(local_times, prepend=local_times[0])
            dt[0] = float(np.median(np.diff(local_times)))   # 첫 프레임 dt(=0) 를 대표 간격으로 보정
        else:
            dt = np.array([0.04], dtype=float)               # 프레임이 1개면 40 ms 로 가정
        phase = 2.0 * np.pi * np.cumsum(motor_rate_hz * dt)
        local_delta = local_amp * np.sin(phase)

        # ⑧ 결과 누적 (episode 는 서로 겹치지 않으므로 += 와 = 모두 안전)
        delta[seg] += local_delta
        active[seg] = np.maximum(active[seg], activity_weight)
        rate_track[seg] = local_rate_track
        extent_track[seg] = local_extent
        conf_track[seg] = local_conf

    # 최종 입 궤적: base + delta 를 [닫힘, 최대 벌림 + overflow] 범위로 클립
    corrected = np.clip(
        base_mouth + delta,
        cfg.mouth_closed_tick,
        cfg.resolved_mouth_open_tick + cfg.resolved_overflow_ticks,
    )
    # roll 보상용 mouth: 같은 떨림을 roll_comp_vibrato_scale 배만 섞음 (헤드 roll 보상이 과하게 흔들리지 않게)
    roll_comp_mouth = np.clip(
        base_mouth + delta * cfg.roll_comp_vibrato_scale,
        cfg.mouth_closed_tick,
        cfg.resolved_mouth_open_tick + cfg.resolved_overflow_ticks,
    )
    return VibratoMotionBundle(
        corrected_mouth=corrected,
        roll_comp_mouth=roll_comp_mouth,
        vibrato_delta=delta,
        vibrato_active=active,
        vibrato_rate_hz=rate_track,
        vibrato_extent_cents=extent_track,
        vibrato_confidence=conf_track,
    )
