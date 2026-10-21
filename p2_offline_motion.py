"""[2단계] 입/헤드 궤적 + 비브라토 — 보컬 wav 하나로 head/mouth CSV 생성.

입력: 분석/<곡>/<곡>_vocals.wav (allin1/demucs 보컬 stem, PCM_16)
출력(run_dir 안):
  <곡>.csv                        head  (roll,pitch,yaw)          40ms/행
  <곡>-delta-big.csv              mouth (비브라토 O)  3열: mouth, head_gesture_ratio, roll_comp_mouth
  <곡>-delta-big_novibrato.csv    mouth (비브라토 X)  같은 3열(3열=base와 동일) — 1·2단계용
  <곡>-delta-big_debug.csv        프레임별 base/delta/corrected/active/rate/extent/conf
  <곡>-delta-big_summary.txt      검출된 비브라토 episode 목록
Pi 는 mouth 의 col0,col1 만 사용. 헤드 궤적은 비브라토와 무관(공통).

원본: offline_pipeline/s3_offline_motion_vibrato.py (= 舊 offline_motion_vibrato.py).
변경점: LED 생성 블록 제거(→ p3_led_allin1), 비브라토 미적용 mouth 추가 저장, run_dir/song_name 인자, dict 반환.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import os
import scipy.io.wavfile as wav
import numpy as np
from scipy import interpolate
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks
import math
import pickle
import librosa
import random
import time

import threading
import tempfile

import soundfile as sf

from pathlib import Path

import p2a_vibrato_detector as analytic
from p2b_vibrato_motion import (
    VibratoMotionConfig,
    apply_vibrato_exaggeration,
    detect_vibrato_episodes,
)

# --------------------------------------------------------------------------------------
# Librosa warmup (first-call latency mitigation)
#
# Observed behavior: the very first call into librosa can stall (backend init / JIT, etc.).
# Goal: make `offline_motion_generation()` feel instantaneous by pre-warming once at
# module import time.
# --------------------------------------------------------------------------------------

_LIBROSA_WARMUP_LOCK = threading.Lock()
_LIBROSA_WARMUP_STARTED = False
_LIBROSA_WARMUP_DONE = threading.Event()


def _librosa_warmup_worker():
    tmp_path = None
    try:
        fs = 16000
        y = np.zeros(fs // 10, dtype=np.float32)
        if y.shape[0] > 0:
            y[::100] = 0.5

        # Trigger common heavy paths (backend init / caches).
        _ = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
            tmp_path = fp.name
        sf.write(tmp_path, y, fs)
        _ = librosa.load(tmp_path, sr=None)
    except Exception:
        # Warmup failures must never break the main pipeline.
        pass
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        _LIBROSA_WARMUP_DONE.set()


def warmup_librosa(*, block: bool = False) -> None:
    """Warm up librosa once.

    - block=False (default): start in a daemon thread and return immediately.
    - block=True: wait until warmup completes.
    """
    global _LIBROSA_WARMUP_STARTED

    with _LIBROSA_WARMUP_LOCK:
        if not _LIBROSA_WARMUP_STARTED:
            _LIBROSA_WARMUP_STARTED = True
            threading.Thread(
                target=_librosa_warmup_worker,
                name="librosa-warmup",
                daemon=True,
            ).start()

    if block:
        _LIBROSA_WARMUP_DONE.wait()


# Start warmup as soon as this module is imported.
warmup_librosa(block=False)


def _create_timestamped_output_dir(base_dir: Path, audio_path: Path) -> tuple[Path, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{audio_path.stem}_{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = base_dir / f"{audio_path.stem}_{timestamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, timestamp


def _save_debug_csv(
    csv_path: Path,
    times_s: np.ndarray,
    base_mouth: np.ndarray,
    corrected_mouth: np.ndarray,
    roll_comp_mouth: np.ndarray,
    vibrato_delta: np.ndarray,
    head_ratio: np.ndarray,
    active: np.ndarray,
    rate_track: np.ndarray,
    extent_track: np.ndarray,
    conf_track: np.ndarray,
) -> None:
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "time_s",
                "base_mouth_tick",
                "vibrato_delta_tick",
                "corrected_mouth_tick",
                "roll_comp_mouth_tick",
                "head_gesture_ratio",
                "vibrato_active",
                "vibrato_rate_hz",
                "vibrato_extent_cents",
                "vibrato_confidence",
            ]
        )
        for row in zip(
            times_s,
            base_mouth,
            vibrato_delta,
            corrected_mouth,
            roll_comp_mouth,
            head_ratio,
            active,
            rate_track,
            extent_track,
            conf_track,
        ):
            writer.writerow([f"{float(v):.8f}" for v in row])


def _save_episode_summary(summary_path: Path, detection_bundle, motion_cfg: VibratoMotionConfig) -> None:
    lines = [
        "Vibrato-enhanced offline motion summary",
        "",
        f"Detected vibrato episodes: {len(detection_bundle.episodes)}",
        f"Vibrato max tick amplitude: {motion_cfg.vibrato_max_ticks:.1f}",
        f"Vibrato extent reference cents: {motion_cfg.vibrato_extent_reference_cents:.1f}",
        f"Fade-in: {motion_cfg.vibrato_fade_in_s:.2f}s",
        "",
    ]
    for i, ep in enumerate(detection_bundle.episodes, start=1):
        lines.append(
            f"{i}. {ep.start_time:.3f}-{ep.end_time:.3f}s | "
            f"rate={ep.final_rate_hz:.3f}Hz | extent={ep.final_extent_cents:.2f}c | "
            f"conf={ep.confidence:.3f} | duration={ep.duration:.3f}s"
        )
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def offline_motion_generation(
    audioname,
    *,
    output_dir="outputs_offline_motion",
    run_dir=None,
    song_name=None,
    vibrato_max_ticks=30.0,
    vibrato_extent_reference=45.0,
    vibrato_fade_in_s=0.12,
):
    """보컬 wav → head/mouth(비브라토 O/X) CSV.

    run_dir 를 주면 그 폴더에 바로 저장(파이프라인용), 없으면 output_dir/<보컬>_<타임스탬프>/ 생성.
    song_name 을 주면 파일명 접두가 그 이름(없으면 보컬 파일명에서 _vocals 제거).
    반환: dict(head, mouth_vibrato, mouth_plain, debug, summary, n_frames, run_dir)
    """
    print('---------------------- NEW MOTION GENERATION STARTED')
    setting_start_time = time.time()
    
    script_root = Path(__file__).resolve().parent
    currentpath = str(script_root)
    audio_path = analytic.resolve_audio_path(str(audioname))
    if run_dir is None:
        output_root = Path(currentpath) / output_dir
        run_dir, timestamp = _create_timestamped_output_dir(output_root, audio_path)
    else:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    motion_name = song_name or (audio_path.stem[:-7] if audio_path.stem.endswith("_vocals") else audio_path.stem)
    audiofilepath = str(audio_path)
    # Match main.cpp csv_control_motor() naming convention exactly:
    #   assets/headMotion/<audioName>.csv
    #   assets/mouthMotion/<audioName>-delta-big.csv
    # When the runtime is invoked as `--csv Pavarotti_vocals`, it looks for
    # `Pavarotti.csv` and `Pavarotti-delta-big.csv`.
    headcsvfilepath = str(run_dir / f'{motion_name}.csv')
    mouthcsvfilepath = str(run_dir / f'{motion_name}-delta-big.csv')
    debugcsvfilepath = run_dir / f'{motion_name}-delta-big_debug.csv'
    summaryfilepath = run_dir / f'{motion_name}-delta-big_summary.txt'
    plainmouthcsvfilepath = str(run_dir / f'{motion_name}-delta-big_novibrato.csv')

    segmentfolder = str(script_root / 'data' / 'segments_offline') + os.sep

    ####################################################################################################################################
    #################################################################################################################################### MOUTH
    ####################################################################################################################################

    max_MOUTH = 0 #default_MOUTH
    min_MOUTH = 250 # 270
    ratio_minOPEN = 0.5

    # Envelope
    peak_distance = 1000 

    # Low pass filter
    num_div = 7 #Larger, smaller fc
    shift = 3

    # max_sigma
    num = 0.8

    # thresholdnum
    thresholdnum = 0.1

    ####################################################################################################################################
    ####################################################################################################################################
    ####################################################################################################################################

    setting_end_time = time.time()
    audio_read_start_time = time.time()
    "## ========================= Audio Read"
    ## https://stackoverflow.com/questions/39316087/how-to-read-a-audio-file-in-python-similar-to-matlab-audioread
    ## https://kr.mathworks.com/help/matlab/ref/audioread.html?lang=en

    fs,y = wav.read(audiofilepath)

    N = y.shape[0]   # number of samples
    Time_sec = (N-1)/fs  # total time (sec)

    y = y / np.iinfo(y.dtype).max  # y range -1 ~ +1

    # xtime = np.linspace(0,Time,N,endpoint = True)
    xtime = np.arange(N)*1/fs + 0

    # ## Plot soundwave
    # plt.figure(0,figsize=(30,5)); plt.grid(True)
    # plt.plot(xtime,y)
    # plt.title('Original soundwave')
    # plt.xlabel('Time (sec)')

    if y.ndim != 1:
        y = y[:, 0]  # use channel 1

    audio_read_end_time = time.time()
    envelope_start_time = time.time()
    "## ======================= Envelope"
    ## Envelope and interpolate
    ## https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
    ## https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html

    interpolate_kind = 'cubic' # 'quadratic' 'linear'

    ## Find high envelope index
    yy = y - np.mean(y)
    high_idx,_ = find_peaks(yy, distance=peak_distance)

    ## Pad with initial and final index
    if high_idx[0] != 0:
        high_idx = np.pad(high_idx,(1,0),'constant',constant_values=0)
    if high_idx[-1] != yy.shape[0]-1:
        high_idx = np.pad(high_idx,(0,1),'constant',constant_values=yy.shape[0]-1)  

    ## Interpolate the local max. points
    f = interpolate.interp1d(xtime[high_idx], yy[high_idx],kind=interpolate_kind) # 'quadratic' 'cubic'
    up = f(xtime)

    envelope_end_time = time.time()
    sampling_start_time = time.time()
    "## ======================= Sampling and Simple Smoothing"

    sampling_interval_ms = 40 # ms

    ## Sampling
    N_size = int(np.floor(Time_sec*1000 / sampling_interval_ms))

    up_sampled = np.zeros(N_size)
    y_sampled = np.zeros(N_size)

    sampling_interval_index = int(np.floor((N-1)/(N_size-1)))
    for i in range(0,N_size):
        up_sampled[i] = up[i*sampling_interval_index]
        y_sampled[i] = y[i*sampling_interval_index]

    ## Simple Smoothing (no negative values)
    soundwave_threshold_zero = 0.05 * np.max(np.abs(y_sampled))
    for i in range(0,y_sampled.shape[0]):
        if abs(y_sampled[i]) < 0.0001:
            up_sampled[i] = 0

        if up_sampled[i] < 0 and abs(y_sampled[i]) > soundwave_threshold_zero:
            up_sampled[i] *= -1
        elif up_sampled[i] < 0:
            up_sampled[i] = 0    

    # t2 = np.arange(0,y_sampled.shape[0])
    # t2 = t2 * sampling_interval_ms / 1000

    # plt.figure(1,figsize=(30,5)); plt.grid(True)
    # plt.plot(t2,y_sampled)
    # plt.plot(t2,up_sampled,'r')
    # plt.title('Sampled y and upper envelope')

    sampling_end_time = time.time()
    envelope_smoothing_start_time = time.time()
    "## ======================= Envelope Smooting"

    ## Remove too large values

    nonzeros_upnew = up_sampled.copy()
    iter = 0
    for i in range(0,up_sampled.shape[0]):
        if abs(y_sampled[i]) > soundwave_threshold_zero:
            if up_sampled[i] != 0:
                nonzeros_upnew[iter] = up_sampled[i]
                iter += 1
    nonzeros_upnew = nonzeros_upnew[0:iter]

    max_Upnew = np.mean(nonzeros_upnew) + num * np.std(nonzeros_upnew)
    up_sampled_maxcut = up_sampled.copy()
    for i in range(0,up_sampled_maxcut.shape[0]):
        if up_sampled_maxcut[i] > max_Upnew:
            up_sampled_maxcut[i] = max_Upnew

    # plt.figure(2,figsize=(30,5)); plt.grid(True)
    # plt.plot(t2,y_sampled)
    # plt.plot(t2,up_sampled_maxcut,'r')
    # plt.title('Maximum value smoothed upper envelope')

    envelope_smoothing_end_time = time.time()
    lowpassfilter_start_time = time.time()
    "## ======================= Low Pass Filter"
    ## Define low pass filter
    ## https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units
    ## https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
    ## https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html

    def butter_lowpass(cutoff, fs, order=5):
        return butter(order, cutoff, fs=fs, btype='low', analog=False)

    def butter_lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    ## Low pass filter
    fc = round(fs / num_div)
    up_sampled_LPF = butter_lowpass_filter(up_sampled_maxcut, fc, fs, order=3)

    ## Compensates for the delay introduced by the filter with shifting
    #up_sampled_LPF = np.zeros(up_sampled_LPF_raw.shape[0])
    #up_sampled_LPF[0:-shift] = up_sampled_LPF_raw[shift:up_sampled_LPF_raw.shape[0]]
    for i in range(0,shift):
        up_sampled_LPF[i-shift] = up_sampled_LPF[-1] 

    ## Noise reduction
    threshold = thresholdnum * np.max(up_sampled_LPF)
    for i in range(0,up_sampled_LPF.shape[0]):
        if up_sampled_LPF[i] < threshold:
            up_sampled_LPF[i] = 0

    # plt.figure(3,figsize=(50,8)); plt.grid(True)
    # plt.plot(t2,y_sampled)
    # plt.plot(t2,up_sampled_LPF,'r')
    # plt.title('Smoothed upper envelope (low pass filter fc={} Hz)'.format(fc))

    lowpassfilter_end_time = time.time()
    minopen_start_time = time.time()
    "## ======================= min_OPEN"
    mouth_raw = up_sampled_LPF.copy()

    min_OPEN = ratio_minOPEN*max(up_sampled_LPF)

    lmin = (np.diff(np.sign(np.diff(up_sampled_LPF))) > 0).nonzero()[0] + 1
    smallpeaks,_ = find_peaks(up_sampled_LPF,height =(None,min_OPEN), prominence = 0.15*max(up_sampled_LPF))

    for i in range(0,smallpeaks.shape[0]):
        lminidx = np.searchsorted(lmin, smallpeaks[i])
        prevMin = lmin[lminidx-1]
        laterMin = lmin[lminidx]

        prev = np.ceil((2*smallpeaks[i]+prevMin)/3)
        later = np.floor((2*smallpeaks[i]+laterMin)/3)
        xx = np.array([prevMin,prev,smallpeaks[i],later,laterMin])

        y = np.array([up_sampled_LPF[prevMin], (5*min_OPEN + up_sampled_LPF[prevMin]) / 6,
                      min_OPEN, (5*min_OPEN + up_sampled_LPF[laterMin]) / 6,up_sampled_LPF[laterMin]])


        try:
            f = interpolate.interp1d(xx, y,kind='quadratic') # 'quadratic' 'cubic'
        except:
            j=1
            while(prev == smallpeaks[i]):
                if up_sampled_LPF[prevMin-1] == 0:
                    prevMin -= 1
                else:
                    prevMin = lmin[lminidx-1-j]
                prev = np.ceil((2*smallpeaks[i]+prevMin)/3)
                j+=1

            k=1
            while(later == smallpeaks[i]):
                if up_sampled_LPF[laterMin+1] == 0:
                    laterMin += 1 
                else:
                    laterMin = lmin[lminidx+k]
                later = np.floor((2*smallpeaks[i]+laterMin)/3)
                k+=1     

            xx = np.array([prevMin,prev,smallpeaks[i],later,laterMin])

            y = np.array([up_sampled_LPF[prevMin], (5*min_OPEN + up_sampled_LPF[prevMin]) / 6,
                          min_OPEN, (5*min_OPEN + up_sampled_LPF[laterMin]) / 6,up_sampled_LPF[laterMin]])

            f = interpolate.interp1d(xx, y,kind='quadratic')

        realtime_x = np.arange(prevMin, laterMin)
        for i in range(0,realtime_x.shape[0]):
            if up_sampled_LPF[realtime_x[i]]<f(realtime_x[i]):
                up_sampled_LPF[realtime_x[i]] = f(realtime_x[i])


    # plt.figure(4,figsize=(100,4)); plt.grid(True);
    # plt.plot(t2,mouth_raw,label='mouth_raw')
    # plt.plot(t2,up_sampled_LPF,'r',label='min_OPEN')
    # plt.title('After Applying Min_open')
    # plt.legend(loc='lower right')

    minopen_end_time = time.time()
    antiq_start_time = time.time()
    "## ======================= Antiq-Sound-Close"
    ## For natural mouth opening (antiq) and closing
    ## https://scienceon.kisti.re.kr/commons/util/originalView.do?cn=JAKO201820540193966&oCn=JAKO201820540193966&dbt=JAKO&journal=NJOU00560240

    ASC = up_sampled_LPF.copy()

    # -0.4: mouth open
    # -0.3: between close~open
    # -0.2: mouth close

    flag_hold_zero = True # hold zero value
    for i in range(0,ASC.shape[0]-1):

        ## When positive sound decreases and becomes zero
        if not flag_hold_zero and up_sampled_LPF[i] == 0:
            ASC[i] = -0.2 # mouth close
            flag_hold_zero = True

        ## When zero sound is about to increase to positive
        if flag_hold_zero and up_sampled_LPF[i] == 0 and up_sampled_LPF[i+1] != 0: 
            if ASC[i] == -0.2: # if mouth close was just signaled from above [if statement] (point zero)
                ASC[i] = -0.3  # then do something between close~open
            else:              # if mouth is currently closed (2 or more consecutive zeros)
                ASC[i] = -0.4  # then open mouth
            flag_hold_zero = False

        ## When there exists positive sound
        if up_sampled_LPF[i] != 0:
            flag_hold_zero = False

    # plt.figure(5,figsize=(30,5)); plt.grid(True)
    # plt.plot(t2,y_sampled,label='voice')
    # plt.plot(t2,up_sampled_LPF,'r',label='upper envelope')
    # plt.stem(t2,ASC,markerfmt='*',linefmt=':',label='ASC')
    # plt.title('Before Applying Antiq-Sound-Close')
    # plt.legend(loc='lower right')

    antiq_end_time = time.time()
    smooth_antiq_start_time = time.time()
    "## ======================= Smooth Antiq, Close"

    ## Smooth closing
    up2mouth = up_sampled_LPF.copy()
    L = up2mouth.shape[0]

    Close = np.zeros(6)

    # ASC[0] == -0.2 -> never happens
    # ASC[1] == -0.2 -> no treatment required

    # ASC[2]~ASC[L-7] == -0.2
    # Mouth close signaled at i
    # Smooth from i-1 to i+4 using value at i-2 (this should be positive)
    for i in range(2,L-6): # -3
        if ASC[i] == -0.2:
            for c in range(0,6): # (0,4)
                Close[c] = up_sampled_LPF[i-2]*(6-c)/7 # (4-c)/5
            for c in range(0,6): # (0,4)
                if Close[c] > up_sampled_LPF[i+c-1]:
                    up2mouth[i+c-1] = Close[c]

    # ASC[L-6]~ASC[L-1] == -0.2
    for i in range(L-6,L): # -3
        if ASC[i] == -0.2:
            for c in range(0,L-i):
                Close[c] = up_sampled_LPF[i-2]*(L-i-c)/(L-i+1)
            for c in range(0,L-i):
                if Close[c] > up_sampled_LPF[i+c-1]:
                    up2mouth[i+c-1] = Close[c]        

    ## Smooth opening
    Antiq = np.zeros(3)

    # ASC[0] == -0.4 -> no treatment required
    # ASC[1] == -0.4: ASC[0:3] 0 0 x -> 0 x/2 x
    if ASC[1] == -0.4:
        up2mouth[1] = up_sampled_LPF[2]/2

    # ASC[2]~ASC[L-2] == -0.4
    # Mouth open signaled at i
    # Smooth from i-2 to i using value at i+1 (this should be positive)
    for i in range(2,L-1):
        if ASC[i] == -0.4:
            for j in range(0,3):
                Antiq[j] = up_sampled_LPF[i+1]*j/3
            for j in range(0,3):
                if Antiq[j] > up2mouth[i-2+j]:
                    up2mouth[i-2+j] = Antiq[j]

    # ASC[L-1] == -0.4 -> never happens

    ## Btw. close~open
    # ASC[0] or ASC[L-1] == -0.3 -> never happens
    for i in range(1,L-1):
        if ASC[i] == -0.3:
            up2mouth[i] = (up2mouth[i-1] + up2mouth[i+1])/3

    # plt.figure(6,figsize=(90,5)); plt.grid(True)
    # plt.plot(t2,y_sampled,label='voice')
    # plt.plot(t2,up_sampled_LPF,'m',label='upper envelope')
    # plt.plot(t2,up2mouth,'r',label='envelope to mouth')
    # plt.title('After Applying Antiq-Sound-Close')
    # plt.legend(loc='lower right')

    smooth_antiq_end_time = time.time()
    scaling_start_time = time.time()
    "## ======================= Scaling to DXL_MOUTH"

    max_up_new = np.max(up2mouth)
    mouth = max_MOUTH*np.ones(up2mouth.shape[0]) - up2mouth*(max_MOUTH-min_MOUTH)/max_up_new


    # plt.figure(7,figsize=(30,5)); plt.grid(True);
    # # plt.plot(t2,mouth_raw,label='mouth_raw')
    # plt.plot(t2,mouth,'r',label='Mouth Motor Input')
    # plt.title('Mouth Motor Input')
    # plt.legend(loc='lower right')

    scaling_end_time = time.time()
    head_gesture_ratio_start_time = time.time()
    "## ======================= head_gesture_ratio"
    # Keep head softly when robot is not singing
    # data : mouth, ratio : ratio of head gesture you want when robot is not singing//0~1

    def find_zero(data, ratio):
        start_rest_idx = []
        rest_len = []
        ratio_remainder = int(10-10*ratio)
        if ratio < 0.5:
            min_len = 2 * ratio_remainder
        else:
            min_len = 10

        count = 1
        head_gesture_ratio = np.ones(N_size)
        # print(head_gesture_ratio)
        head_gesture_ratio_idx = (np.where(data == max_MOUTH)[0])
        for i in range(head_gesture_ratio_idx.shape[0]-1):
            if head_gesture_ratio_idx[i + 1] == head_gesture_ratio_idx[i]+1:
                count += 1
            else:
                start_rest_idx.append(head_gesture_ratio_idx[i] - count + 1)
                rest_len.append(count)
                count = 1
        # print(start_rest_idx)
        # print(rest_len)
        for i in range(len(rest_len)):
            if rest_len[i] >= min_len:
                for j in range(ratio_remainder):
                    head_gesture_ratio[start_rest_idx[i]+j] = 0.9 -j *0.1
                if rest_len[i] > 2*ratio_remainder:
                    for j in range(ratio_remainder,rest_len[i]-ratio_remainder):
                        head_gesture_ratio[start_rest_idx[i]+j] = ratio
                k = 0
                for j in range(rest_len[i]-ratio_remainder,rest_len[i]):
                    head_gesture_ratio[start_rest_idx[i]+j] = ratio + k *0.1
                    k += 1
        return head_gesture_ratio

    head_gesture_ratio = find_zero(mouth,0.6)
    # print(head_gesture_ratio[500:1000])

    ####################################################################################################################################
    #################################################################################################################################### VIBRATO
    ####################################################################################################################################

    vibrato_detection_start_time = time.time()
    times_s = np.arange(mouth.shape[0]) * sampling_interval_ms / 1000.0
    base_mouth = mouth.copy()
    detection = detect_vibrato_episodes(audio_path)
    vibrato_detection_end_time = time.time()

    vibrato_apply_start_time = time.time()
    motion_cfg = VibratoMotionConfig(
        mouth_closed_tick=max_MOUTH,
        mouth_open_tick=min_MOUTH,
        vibrato_max_ticks=vibrato_max_ticks,
        vibrato_extent_reference_cents=vibrato_extent_reference,
        vibrato_fade_in_s=vibrato_fade_in_s,
    )
    vibrato_motion = apply_vibrato_exaggeration(times_s, base_mouth, detection, motion_cfg)
    mouth = vibrato_motion.corrected_mouth
    vibrato_apply_end_time = time.time()

    ####################################################################################################################################
    ####################################################################################################################################
    ####################################################################################################################################

    output_mouth_ratio = np.column_stack(
        (mouth, head_gesture_ratio, vibrato_motion.roll_comp_mouth)
    )
    np.savetxt(mouthcsvfilepath,output_mouth_ratio, fmt='%.10e', delimiter=',')
    # 비브라토 미적용(원본 오프라인) 입 궤적 — 1·2단계 출력용.
    # 열 구성은 비브라토판과 동일(3열)하되 3열(roll 보상)은 비브라토가 없으니 base 와 동일.
    output_mouth_plain = np.column_stack((base_mouth, head_gesture_ratio, base_mouth))
    np.savetxt(plainmouthcsvfilepath, output_mouth_plain, fmt='%.10e', delimiter=',')
    _save_debug_csv(
        debugcsvfilepath,
        times_s,
        base_mouth,
        mouth,
        vibrato_motion.roll_comp_mouth,
        vibrato_motion.vibrato_delta,
        head_gesture_ratio,
        vibrato_motion.vibrato_active,
        vibrato_motion.vibrato_rate_hz,
        vibrato_motion.vibrato_extent_cents,
        vibrato_motion.vibrato_confidence,
    )
    _save_episode_summary(summaryfilepath, detection, motion_cfg)
    print('---------------------- NEW MOUTH MOTION GENERATED')

    # (LED 생성 블록은 제거됨 — back 조명은 p3_led_allin1 이 allin1 설계도로 만든다)

    head_gesture_ratio_end_time = time.time()

    ####################################################################################################################################
    #################################################################################################################################### HEAD
    ####################################################################################################################################

    head_motion_ready_start_time = time.time()

    dt = 40
    segmentLength = 360          # ms
    shift = 80                   # ms (overlap = segmentLength - shift)
    L = round(segmentLength/dt)  # length
    l = round(shift/dt)          # shift

    classNum = 4

    increaseAmp = 1.5
    rpymax = [0.7,0.6,1.0]

    audiofeaturewindow = 40 # ms
    audiofeatureoverlap = 23.3 # ms

    rpy_grad = 'one2one'
    audio_grad = 'one2one'

    ################################################################################################################################# BASIC

    def makeXtime(Time_ms,feature):

        if type(feature) == int:
            return np.round(np.linspace(0, Time_ms, num=feature, endpoint=True)) # ms
        elif type(feature) == np.ndarray:
            return np.round(np.linspace(0, Time_ms, num=feature.shape[0], endpoint=True)) # ms

    ################################################################################################################################# AUDIO FEATURES

    def tmp_syncAudioVideoOneFeature(xtime_audiofeature,xtime_video,audiofeature1,delta=15):

        audiofeature_sync = np.zeros(xtime_video.shape[0])

        for i in range(audiofeature_sync.shape[0]):
            for j in range(xtime_audiofeature.shape[0]):
                flag_found = False
                if abs(xtime_video[i]-xtime_audiofeature[j]) < delta:
                    audiofeature_sync[i] = audiofeature1[j]
                    flag_found = True
                    break
            if not flag_found:
                print("error")

        return audiofeature_sync   

    def syncAudioVideo(xtime_audiofeature,xtime_video,audiofeature,delta=15):

        if audiofeature.ndim == 1:
            return tmp_syncAudioVideoOneFeature(xtime_audiofeature,xtime_video,audiofeature,delta)
        else:
            audiofeature_sync = np.zeros([xtime_video.shape[0],audiofeature.shape[1]])
            for i in range(audiofeature.shape[1]):
                audiofeature_sync[:,i] = tmp_syncAudioVideoOneFeature(xtime_audiofeature,xtime_video,audiofeature[:,i],delta)#.reshape(-1,1)

            return audiofeature_sync     

    def MeanFiltering(original,filtersize=1,shift=0):    

        Time = original.shape[0]

        filtered = original.copy()

        if original.ndim != 1:

            iteration = 0
            for i in range(Time):
                if iteration < filtersize-1:
                    # filtered[i,:] = np.round(np.mean(original[i-iteration:i+1,:],axis=0)) # mean of rows (column direction)
                    filtered[i,:] = np.mean(original[i-iteration:i+1,:],axis=0) # mean of rows (column direction)
                else:
                    # filtered[i,:] = np.round(np.mean(original[i-(filtersize-1):i+1,:],axis=0))
                    filtered[i,:] = np.mean(original[i-(filtersize-1):i+1,:],axis=0)

                iteration = iteration + 1

        else:

            iteration = 0
            for i in range(Time):
                if iteration < filtersize-1:                
                    filtered[i] = np.mean(original[i-iteration:i+1],axis=0) # mean of rows (column direction)
                else:
                    # filtered[i,:] = np.round(np.mean(original[i-(filtersize-1):i+1,:],axis=0))
                    filtered[i] = np.mean(original[i-(filtersize-1):i+1],axis=0)

                iteration = iteration + 1

        filtered[0:Time-shift] = filtered[shift:]

        return filtered

    def makeNoVoiceSteps(voiced_flag_sync):
        novoicesteps = np.zeros(voiced_flag_sync.shape[0])

        novoice = 0
        for i in range(novoicesteps.shape[0]):
            if voiced_flag_sync[i] == 0:
                novoice += 1
            elif voiced_flag_sync[i] == 1:
                novoice = 0

            novoicesteps[i] = novoice

        return novoicesteps

    def makeNoVoiceExp(energy,thr=0.01,div=10):

        voiceflag_e = energy.copy()
        voiceflag_e[voiceflag_e< thr]=0
        voiceflag_e[voiceflag_e>=thr]=1
        novoicesteps = makeNoVoiceSteps(voiceflag_e)
        novoiceexp = np.exp(-novoicesteps/div)
        novoiceexp[novoiceexp<=0.3]=0.3

        # novoiceexp = np.r_[ novoiceexp , np.array([novoiceexp[-1]]*100) ]

        return novoiceexp,voiceflag_e

    ################################################################################################################################# 

    def getSegmentAverageGrad(y,delta='one2one',mode='abs'):

        x = y.copy()
        if x.ndim == 1:
            x = x.reshape(-1,1)

        if delta == 'one2one':
            grad = x[1:,:]-x[:-1,:] # T-1 x dim
        elif delta == 'end2end':
            grad = x[-1,:]-x[0,:] # dim
        else:
            print('getSegmentAverageGrad delta error')

        if mode == 'abs':
            grad = np.abs(grad)
        elif mode == 'pos':
            grad = grad[grad>0]
            # grad[grad<0] = 0
        elif mode == 'neg':
            grad = grad[grad<0]
            # grad[grad>0] = 0
        elif mode == 'org':
            pass
        else:
            print('getSegmentAverageGrad mode error')

        if grad.size==0:
            return 0

        return np.average(grad)

    def assignClassWith1DMiddleBoundary(x,boundaries):
        # boundaries: no min,max; only middle boundaries; ex) ~1,1~2,2~3,3~ -> 1,2,3
        for i in range(len(boundaries)-1):
            if x < boundaries[i]:
                return i
            elif x < boundaries[i+1]:
                return i+1
        return len(boundaries)

    ################################################################################################################################# HEAD MOTION GENERATION

    def getNextSegment_PointSeg(PrevEnd,rpysegmentsClass,gradient=True):

        # PrevEnd: 3(rpy)
        # rpysegmentsClass: L x 3(rpy) x numberofsegments

        distselectnum = 20
        distselectdist = 0.2
        randomchoosenum = 5 # < distselectnum
        # randomchoosenum = 1 # < distselectnum

        distselectnum = min(distselectnum,rpysegmentsClass.shape[2]) # rpysegmentsClass.shape[2] = len(dists)
        randidx = random.randint(0,randomchoosenum-1)

        startpoints = rpysegmentsClass[0,:,:].T
        endpoint_tile = np.tile(PrevEnd.reshape(1,-1),[rpysegmentsClass.shape[2],1])

        dists = np.linalg.norm(endpoint_tile - startpoints,axis=1)

        ## dists[minsortedindex[0]] : minimum distance
        ## rpysegmentsClass[:,:,minsortedindex[0]] : segment with minimum distance     
        distminsortedindex = np.array( sorted(range(len(dists)), key=lambda k: dists[k]) )[:distselectnum]
        distusingindices = distminsortedindex[ dists[distminsortedindex] < distselectdist ] # dists[distusingindices]: 최소 거리 순

        if len(distusingindices) < randomchoosenum:
            index = np.argmin(dists)
            return rpysegmentsClass[:,:,index]

        # min gradient
        if gradient:  
            firsttwopoints = rpysegmentsClass[:2,:,distusingindices]      # shape: 2 x 3(rpy) x distselectnum; 거리 짧은 순으로 계산됨
            startgrads = firsttwopoints[1,:,:] - firsttwopoints[0,:,:]    # shape: 3(rpy) x distselectnum
            gradnorms = np.linalg.norm(startgrads,axis=0)                 # shape: distselectnum
            gradminsortedindex = sorted(range(len(gradnorms)), key=lambda k: gradnorms[k])
            index = distusingindices[gradminsortedindex[randidx]]        
        else:
            index = distusingindices[randidx]          

        return rpysegmentsClass[:,:,index]

    def getNextSegment_SegSeg(PrevEndOneBefore,PrevEnd,rpysegmentsClass,gradient=True,gotozero=True):

        # PrevEndOneBefore: 3(rpy)
        # PrevEnd: 3(rpy)
        # rpysegmentsClass: L x 3(rpy) x numberofsegments

        distselectnum = 20 
        distselectdist = 0.2
        gradselectnum = 15 # < distselectnum
        randomchoosenum = 10 # < distselectnum,gradselectnum
        # randomchoosenum = 1 # < distselectnum,gradselectnum

        distselectnum = min(distselectnum,rpysegmentsClass.shape[2]) # rpysegmentsClass.shape[2] = len(dists)
        randidx = random.randint(0,randomchoosenum-1) # 0 ~ randomchoosenum-1

        startpoints = rpysegmentsClass[0,:,:].T
        endpoint_tile = np.tile(PrevEnd.reshape(1,-1),[rpysegmentsClass.shape[2],1])

        dists = np.linalg.norm(endpoint_tile - startpoints,axis=1)

        ## dists[minsortedindex[0]] : minimum distance
        ## rpysegmentsClass[:,:,minsortedindex[0]] : segment with minimum distance     
        distminsortedindex = np.array( sorted(range(len(dists)), key=lambda k: dists[k]) )[:distselectnum]        
        distusingindices = distminsortedindex[ dists[distminsortedindex] < distselectdist ]

        if len(distusingindices) < randomchoosenum:
            index = np.argmin(dists)
            return rpysegmentsClass[:,:,index]

        # min gradient difference
        if gradient:

            gradselectnum = min(gradselectnum,len(distusingindices))

            firsttwopoints = rpysegmentsClass[:2,:,distusingindices]             # shape: 2 x 3(rpy) x distselectnum
            startgrads = firsttwopoints[1,:,:] - firsttwopoints[0,:,:]       # shape: 3(rpy) x distselectnum

            gradfinal = PrevEnd - PrevEndOneBefore
            gradfinal_tile = np.tile(gradfinal.reshape(-1,1),[1,len(distusingindices)]) # shape: 3(rpy) x distselectnum

            graddists = np.linalg.norm(gradfinal_tile - startgrads,axis=0)   # shape: distselectnum            
            gradminsortedindex = sorted(range(len(graddists)), key=lambda k: graddists[k])[:gradselectnum]
            gradusingindices = np.array(distusingindices)[gradminsortedindex] # dist 작은 것들 grad 작은 순으로 정렬한 index

            # go to zero - weighted scores
            if gotozero:

                tmp_segments = rpysegmentsClass[:,:,gradusingindices] # dist, grad 조건 충족한 segment들 (일정 dist 이하들을 grad 작은 순으로 정렬); L x 3 x gradselectnum            
                tmp_rpyscores = np.multiply(tmp_segments[0,:,:],tmp_segments[-1,:,:] - tmp_segments[0,:,:]) # shape 3 x gradselectnum            
                gotozeroscores = np.sum(tmp_rpyscores,axis=0) # shape gradselectnum            

                scoreminsortedindex = sorted(range(len(gotozeroscores)), key=lambda k: gotozeroscores[k])

                # mm+randidx < gradselectnum-1
                mm = 0
                index = gradusingindices[scoreminsortedindex[mm+randidx]]            
            else:
                index = gradusingindices[randidx]
        else:
            index = distusingindices[randidx]    

        return rpysegmentsClass[:,:,index]

    def ConnectPointAndSegment(point,test2,n_interpolate=5,n_new=4):

        # 1 from point, n_interpolate from test2 -> 1 + n_interpolate used for interpolation
        # n_interpolate must be >= 3
        # n_new new points

        point = point.copy()
        test2 = test2.copy()

        nn = n_interpolate + n_new    

        if nn > test2.shape[0]:
            print('point-segment interpolation warning')
            k = 0
            while nn > test2.shape[0]:
                if k%2 == 0 and n_new > 1:
                    n_new -= 1                
                elif k%2 == 1 and n_interpolate >= 4:                
                    n_interpolate -= 1                
                nn = n_interpolate + n_new                
                k += 1
                if k > 200:
                    print('point-segment interpolation failed')
                    return np.append(point.reshape(1,-1),test2,axis=0)

        x = list(range(nn+1))

        for rpy in range(3):
            y = [point[rpy]] + test2[:nn,rpy].tolist()

            x_interpolate = [x[0]] + x[-n_interpolate:]
            y_interpolate = [y[0]] + y[-n_interpolate:]
            tck = interpolate.splrep(x_interpolate, y_interpolate)

            y_new = [y[0]]
            for i in range(1,1+n_new):
                y_new += [interpolate.splev(i, tck)]
            y_new += y[-n_interpolate:]

            test2[:nn,rpy] = np.array(y_new[-nn:])

        return np.append(point.reshape(1,-1),test2,axis=0)

    def ConnectTwoSegments(test1,test2,n_interpolate=3,n_new=4):

        # n_interpolate from test1, n_interpolate from test2 -> 2*n_interpolate used for interpolation
        # n_interpolate must be >= 2
        # 2*n_new new points

        test1 = test1.copy()
        test2 = test2.copy()

        nn = n_interpolate + n_new

        if nn > test2.shape[0]:
            print('segment-segment interpolation warning')
            k = 0
            while nn > test2.shape[0]:
                if k%2 == 0 and n_new > 1:
                    n_new -= 1
                elif k%2 == 1 and n_interpolate >= 3:                
                    n_interpolate -= 1                
                nn = n_interpolate + n_new  
                k += 1
                if k > 100:
                    print('segment-segment interpolation failed')
                    return np.append(test1,test2,axis=0)

        x = list(range(nn*2))

        for rpy in range(3):
            y = test1[-nn:,rpy].tolist() + test2[:nn,rpy].tolist()

            x_interpolate = x[0:n_interpolate] + x[-n_interpolate:]        
            y_interpolate = y[0:n_interpolate] + y[-n_interpolate:]
            tck = interpolate.splrep(x_interpolate, y_interpolate)

            y_new = y[0:n_interpolate]
            for i in range(n_interpolate,n_interpolate+2*n_new):
                y_new += [interpolate.splev(i, tck)]
            y_new += y[-n_interpolate:]

            test1[-nn:,rpy] = np.array(y_new[:nn])
            test2[:nn,rpy] = np.array(y_new[-nn:])

        return np.append(test1,test2,axis=0)

    def multExpToSegment(novoiceexp_seg,NextSegment):

        delta = NextSegment[1:,:] - NextSegment[:-1,:]
        newNextSegment = NextSegment.copy()

        for i in range(1,newNextSegment.shape[0]):
            # newNextSegment[i,:] = newNextSegment[i-1,:] + novoiceexp_seg[i-1] * delta[i-1,:] # onset delayed one step
            newNextSegment[i,:] = newNextSegment[i-1,:] + novoiceexp_seg[i] * delta[i-1,:] # onset not delayed

        return newNextSegment

    def wav2rpy(wavpath,audiofeaturewindow,audiofeatureoverlap,rpysegments_parallel,e_bd,div=10):

        head_1_start_time = time.time()
        librosa_load_start_time = time.time()

        y, fs = librosa.load(wavpath, sr=None)    
        # y, fs = sf.read(wavpath, dtype='float32')
        # if len(y.shape) > 1:  # 스테레오일 경우 모노로 변환
        #     y = np.mean(y, axis=1)

        Time_ms = 1000*(y.shape[0]-1)/fs 
        
        librosa_load_end_time = time.time()

        # print('Length of audio file = {:.2f} seconds'.format(Time_ms/1000))

        librosa_rms_start_time = time.time()
        energy = librosa.feature.rms(y=y , frame_length=round(audiofeaturewindow/1000*fs) , hop_length=round(audiofeatureoverlap/1000*fs) )[0]
        librosa_rms_end_time = time.time()
        meanfiltering_start_time = time.time()

        energy = MeanFiltering(energy,5,2) # mean filtering

        meanfiltering_end_time = time.time()
        makextime_start_time = time.time()

        xtime_rpy = makeXtime(Time_ms,math.floor(Time_ms/dt))
        xtime_audio = makeXtime(Time_ms,energy)

        makextime_end_time = time.time()
        sync_start_time = time.time()

        energy = syncAudioVideo(xtime_audio,xtime_rpy,energy,delta=13)

        L = rpysegments_parallel[0].shape[0]

        sync_end_time = time.time()
        novoiceexp_start_time = time.time()

        novoiceexp = makeNoVoiceExp(energy,thr=0.01,div=div)[0]
        novoiceexp = np.r_[ novoiceexp , np.array([novoiceexp[-1]]*100) ]

        rpy_generated = np.empty([0,3])

        novoiceexp_end_time = time.time()
        head_1_end_time = time.time()
        head_2_start_time = time.time()

        i = 0
        e_delta_val_sq = []
        while i+L < energy.shape[0]:
            getgrad_start_time = time.time()

            e_seg = energy[i:i+L]
            e_delta_val = getSegmentAverageGrad(e_seg,delta=audio_grad,mode='abs')

            e_delta_val_sq += [e_delta_val] * e_seg.shape[0]

            getgrad_end_time = time.time()
            assignclass_start_time = time.time()

            segClass = 0 if i==0 else assignClassWith1DMiddleBoundary(e_delta_val,e_bd)

            # print(segClass)

            novoiceexp_seg = novoiceexp[i:i+L]

            assignclass_end_time = time.time()

            if i == 0:
                getnextsegment_start_time = time.time()

                NextSegment = getNextSegment_PointSeg( np.zeros(3) , rpysegments_parallel[segClass] , gradient=True )

                getnextsegment_end_time = time.time()
                multexp_start_time = time.time()

                NextSegment = multExpToSegment(novoiceexp_seg,NextSegment)
                
                multexp_end_time = time.time()
                connectpointandsegment_start_time = time.time()

                rpy_generated = ConnectPointAndSegment( np.zeros(3) , NextSegment , n_interpolate=5 , n_new=3 )

                connectpointandsegment_end_time = time.time()

                # print("Get Segment Average Grad Time: {:.3f} seconds".format(getgrad_end_time - getgrad_start_time))
                # print("Assign Class Time: {:.3f} seconds".format(assignclass_end_time - assignclass_start_time))
                # print("Get Next Segment Time: {:.3f} seconds".format(getnextsegment_end_time - getnextsegment_start_time))
                # print("Multiply Exp Time: {:.3f} seconds".format(multexp_end_time - multexp_start_time))
                # print("Connect Point And Segment Time: {:.3f} seconds".format(connectpointandsegment_end_time - connectpointandsegment_start_time))
            else:
                getnextsegment_e_start_time = time.time()
                NextSegment = getNextSegment_SegSeg( rpy_generated[-2,:] , rpy_generated[-1,:] , rpysegments_parallel[segClass] , gradient=True , gotozero=True )
                getnextsegment_e_end_time = time.time()
                multexp_e_start_time = time.time()
                NextSegment = multExpToSegment(novoiceexp_seg,NextSegment)
                multexp_e_end_time = time.time()
                connecttwosegments_start_time = time.time()
                rpy_generated = ConnectTwoSegments( rpy_generated , NextSegment , n_interpolate=3 , n_new=4 ) 
                connecttwosegments_end_time = time.time()           

            i += L
        
        head_2_end_time = time.time()
        head_3_start_time = time.time()

        NextSegment = getNextSegment_SegSeg( rpy_generated[-2,:] , rpy_generated[-1,:] , rpysegments_parallel[3] , gradient=True , gotozero=True )    
        NextSegment = multExpToSegment(novoiceexp[i:i+NextSegment.shape[0]],NextSegment)

        rpy_generated = ConnectTwoSegments( rpy_generated , NextSegment , n_interpolate=3 , n_new=4 )    
        rpy_generated = rpy_generated[:energy.shape[0],:]

        e_delta_val_sq += [0] * (rpy_generated.shape[0] - len(e_delta_val_sq))

        head_3_end_time = time.time()
        
        # print("Head Motion 1 Time: {:.3f} seconds".format(head_1_end_time - head_1_start_time))
        # print(" - Librosa Load Time: {:.3f} seconds".format(librosa_load_end_time - librosa_load_start_time))
        # print(" - Librosa RMS Time: {:.3f} seconds".format(librosa_rms_end_time - librosa_rms_start_time))
        # print(" - Mean Filtering Time: {:.3f} seconds".format(meanfiltering_end_time - meanfiltering_start_time))
        # print(" - Make Xtime Time: {:.3f} seconds".format(makextime_end_time - makextime_start_time))
        # print(" - Sync Time: {:.3f} seconds".format(sync_end_time - sync_start_time))
        # print(" - No Voice Exp Time: {:.3f} seconds".format(novoiceexp_end_time - novoiceexp_start_time))
        # print("Head Motion 2 Time: {:.3f} seconds".format(head_2_end_time - head_2_start_time))
        # print(" - Get Next Segment Time: {:.3f} seconds".format(getnextsegment_e_end_time - getnextsegment_e_start_time))
        # print(" - Multiply Exp Time: {:.3f} seconds".format(multexp_e_end_time - multexp_e_start_time))
        # print(" - Connect Two Segments Time: {:.3f} seconds".format(connecttwosegments_end_time - connecttwosegments_start_time))
        # print("Head Motion 3 Time: {:.3f} seconds".format(head_3_end_time - head_3_start_time))

        return rpy_generated, xtime_rpy , energy , e_delta_val_sq

    ####################################################################################################################################
    ####################################################################################################################################
    ####################################################################################################################################

    rpysegmentsavepath = segmentfolder+'rpysegments_'+str(segmentLength) + '_' + rpy_grad +   '_'+str(classNum)+      '.npy'
    audioboundsavepath = segmentfolder+'energybd_'+str(segmentLength) + '_' + audio_grad +    '_'+str(classNum)+     '.npy'

    with open(rpysegmentsavepath, 'rb') as f:
        rpysegments_parallel = pickle.load(f)
    with open(audioboundsavepath,'rb') as f:
        e_bd = pickle.load(f)

    head_motion_ready_end_time = time.time()
    wav2rpy_start_time = time.time()

    ####################################################################################################################################
    ####################################################################################################################################
    ####################################################################################################################################

    rpy_generated = wav2rpy( audiofilepath , audiofeaturewindow , audiofeatureoverlap , rpysegments_parallel , e_bd , div=5 )[0]

    wav2rpy_end_time = time.time()
    head_motion_generation_start_time = time.time()

    rpy_generated_delta = np.zeros(rpy_generated.shape)
    rpy_generated_delta[1:,:] = rpy_generated[1:,:] - rpy_generated[:-1,:]

    np.savetxt(headcsvfilepath, rpy_generated, delimiter=",",fmt='%f')

    print('---------------------- NEW HEAD MOTION GENERATED')
    head_motion_generation_end_time = time.time()
    head_saved = True

    # 전체 시간 출력
    # print("Setting Time: {:.2f} seconds".format(setting_end_time - setting_start_time))
    # print("Audio Read Time: {:.2f} seconds".format(audio_read_end_time - audio_read_start_time))
    # print("Envelope Extraction Time: {:.2f} seconds".format(envelope_end_time - envelope_start_time))
    # print("Sampling Time: {:.2f} seconds".format(sampling_end_time - sampling_start_time))
    # print("Envelope Smoothing Time: {:.2f} seconds".format(envelope_smoothing_end_time - envelope_smoothing_start_time))
    # print("Low Pass Filter Time: {:.2f} seconds".format(lowpassfilter_end_time - lowpassfilter_start_time))
    # print("Min_OPEN Time: {:.2f} seconds".format(minopen_end_time - minopen_start_time))
    # print("Antiq-Sound-Close Time: {:.2f} seconds".format(antiq_end_time - antiq_start_time))
    # print("Smooth Antiq, Close Time: {:.2f} seconds".format(smooth_antiq_end_time - smooth_antiq_start_time))
    # print("Scaling to DXL_MOUTH Time: {:.2f} seconds".format(scaling_end_time - scaling_start_time))
    # print("Head Gesture Ratio Time: {:.2f} seconds".format(head_gesture_ratio_end_time - head_gesture_ratio_start_time))
    # print("Vibrato Detection Time: {:.2f} seconds".format(vibrato_detection_end_time - vibrato_detection_start_time))
    # print("Vibrato Apply Time: {:.2f} seconds".format(vibrato_apply_end_time - vibrato_apply_start_time))
    # print("Head Motion Ready Time: {:.2f} seconds".format(head_motion_ready_end_time - head_motion_ready_start_time))
    # print("wav2rpy Time: {:.2f} seconds".format(wav2rpy_end_time - wav2rpy_start_time))
    # print("Head Motion Generation Time: {:.2f} seconds".format(head_motion_generation_end_time - head_motion_generation_start_time))

    print("Saved mouth csv (vibrato): {}".format(mouthcsvfilepath))
    print("Saved mouth csv (plain)  : {}".format(plainmouthcsvfilepath))
    print("Saved head csv: {}".format(headcsvfilepath))
    print("Saved debug csv: {}".format(debugcsvfilepath))
    print("Saved summary txt: {}".format(summaryfilepath))
    print("Offline Motion Generation Total Time: {:.2f} seconds".format(head_motion_generation_end_time - setting_start_time))
    return {
        "head": Path(headcsvfilepath),
        "mouth_vibrato": Path(mouthcsvfilepath),
        "mouth_plain": Path(plainmouthcsvfilepath),
        "debug": Path(debugcsvfilepath),
        "summary": Path(summaryfilepath),
        "n_frames": int(mouth.shape[0]),
        "run_dir": Path(run_dir),
    }


CLI_PARSER = argparse.ArgumentParser(description="[2단계] 보컬 wav → head/mouth(+비브라토) 궤적 (단독 실행용)")
CLI_PARSER.add_argument("audio_path", help="보컬 wav 경로 (예: 분석/Butter_BTS/Butter_BTS_vocals.wav)")
CLI_PARSER.add_argument("--output-dir", default="outputs_offline_motion")
CLI_PARSER.add_argument("--vibrato-max-ticks", type=float, default=30.0)
CLI_PARSER.add_argument("--vibrato-extent-reference", type=float, default=45.0)
CLI_PARSER.add_argument("--vibrato-fade-in-s", type=float, default=0.12)

if __name__ == "__main__":
    # 전체 실행시간 측정 (초 단위)
    
    for i in range(1):
        print("\n===== Run {} =====".format(i+1))
        start_time = time.time()
        cli_args = CLI_PARSER.parse_args()
        offline_motion_generation(
            cli_args.audio_path,
            output_dir=cli_args.output_dir,
            vibrato_max_ticks=cli_args.vibrato_max_ticks,
            vibrato_extent_reference=cli_args.vibrato_extent_reference,
            vibrato_fade_in_s=cli_args.vibrato_fade_in_s,
        )
        end_time = time.time()
        print("총 실행시간: {:.2f} seconds".format(end_time - start_time))
