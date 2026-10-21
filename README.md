# Vibrato 궤적 생성

원곡 wav → allin1 분리 → 입/헤드 궤적 → back 조명(LED) → 비브라토. 곡마다 **3단계 궤적 세트**를 만든다.

```
1_오프라인궤적/          head + mouth                 (조명 X, 비브라토 X)
2_오프라인궤적+조명/     head + mouth + led           (비브라토 X)
3_전체_비브라토포함/      head + mouth(비브라토) + led  ← 로봇 재생 기본
```

## 설치

1. Python 3.11 환경 (Windows)
   ```
   python -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
2. allin1 환경 (WSL) — 새 곡을 분석할 때만 필요. 이미 `분석/<곡>/` 이 있는 곡은 없어도 된다.
   allin1 이 의존하는 natten 은 Windows 휠이 없어 WSL Ubuntu 안의 venv(`/root/allin1env`)에서 돈다.
   1. WSL + Ubuntu 22.04 설치 (관리자 PowerShell, PC 마다 1회): `wsl --install -d Ubuntu-22.04` → 재부팅 → Ubuntu 첫 실행에서 사용자 생성.
      GPU 확인: `wsl -d Ubuntu-22.04 -u root -- nvidia-smi` (Windows NVIDIA 드라이버가 그대로 보이면 OK)
   2. `wsl_setup\setup_allin1env.bat` 실행 (약 3GB 다운로드, 15~30분). 끝에 `SETUP_DONE`, `WSL_ALLIN1_OK` 가 찍히면 성공.
      - 배포판 이름이 다르면 `setup_allin1env.bat <이름>` 으로 넘기고, 실행 전에 환경변수 `VIBRATO_WSL_DISTRO=<이름>` 을 준다.
      - 버전 고정: torch 2.2.0+cu121 + natten 0.15.1. allin1 1.1.0 은 natten 0.14 시절 API 를 import 하므로 natten 0.17.x 는 안 된다.
      - 사내망 SSL 인터셉트가 있으면 그 루트 인증서를 `wsl_setup\eprism_ssl_root.crt` 로 두면 [2/7] 단계에서 WSL 신뢰소에 등록한다(없으면 건너뜀, git 에는 안 올라감).

## 실행

`music/<곡>.wav` 에 원곡을 넣고:

```
.venv\Scripts\python.exe run.py                       # music/ 전곡 (완료된 곡은 건너뜀)
.venv\Scripts\python.exe run.py Butter_BTS            # 한 곡
.venv\Scripts\python.exe run.py Butter_BTS --force    # 궤적 다시 (allin1 캐시는 재사용)
.venv\Scripts\python.exe run.py Butter_BTS --refresh  # allin1 재분석까지 전부 다시
.venv\Scripts\python.exe run.py Butter_BTS --led-only # 모션 재사용, LED 만 다시 (튜닝 반복)
```

- allin1 분리+구조분석은 GPU(RTX 3050) 기준 3분 곡에 약 1분. 결과는 `분석/<곡>/` 에 남아 2회차부터 즉시.
- 나머지 단계(모션+LED)는 곡당 1~2분, pyin 캐시가 생긴 뒤에는 ~20초.
- 헤드 궤적은 세그먼트 라이브러리에서 무작위로 고르므로 실행마다 조금씩 달라진다. 입/LED 는 같은 입력이면 같다.

## 폴더 구조

```
├─ run.py                  ★ 진입점
├─ common.py               폴더 규칙·경로 헬퍼 (모든 단계가 이것만 봄)
├─ p1_allin1_separate.py   [1] allin1 구조분석 + demucs 4-stem 분리 (WSL 호출) → 분석/<곡>/
├─ p2_offline_motion.py    [2] 입/헤드 궤적 + 비브라토
├─ p2a_vibrato_detector.py     비브라토 검출 엔진
├─ p2b_vibrato_motion.py       검출 결과 → 입 모터 delta 변환 (VibratoMotionConfig)
├─ p3_led_allin1.py        [3] back 조명(LED) 밝기 — allin1 라벨·박자 기반
├─ data/segments_offline/  헤드 세그먼트 라이브러리(.npy) — 헤드 궤적 생성에 필수
├─ wsl_setup/              allin1 WSL 환경 설치 스크립트
│
│  (아래는 git 에 안 올라감)
├─ music/                  원곡 wav 넣는 곳 (music/raw/ 도 인식)
├─ data/pyin_cache/        pyin f0 캐시 (자동 생성)
├─ 분석/<곡>/               allin1 json · stems/{vocals,drums,bass,other}.wav · <곡>_vocals.wav · <곡>_MR.wav · 요약 txt
└─ 궤적/<곡>/
    ├─ 1_오프라인궤적/          <곡>.csv  <곡>-delta-big.csv
    ├─ 2_오프라인궤적+조명/     <곡>.csv  <곡>-delta-big.csv  <곡>-led.csv
    ├─ 3_전체_비브라토포함/      <곡>.csv  <곡>-delta-big.csv(비브라토)  <곡>-led.csv
    ├─ _중간산출물/             원본 산출물 + _novibrato / _debug csv / 비브라토·LED 요약 txt
    └─ _manifest.json          생성 정보(행 수, BPM, 비브라토 파라미터, 단계 설명)
```

## 단계별 처리 흐름

| 단계 | 모듈 | 입력 → 출력 |
|---|---|---|
| [1] 분리 | `p1_allin1_separate` | 원곡 → allin1(bpm/beats/downbeats/segments) + demucs 4-stem. vocals=stem 복사, MR=drums+bass+other 합 |
| [2] 궤적 | `p2_offline_motion` | vocals → head(roll,pitch,yaw), mouth 두 벌(비브라토 O/X) |
| [3] 조명 | `p3_led_allin1` | allin1 라벨·박자 + drums/반주 stem → LED 밝기(입/헤드와 같은 행 수로 정렬) |
| [4] 정리 | `run.py` | 1·2·3 단계 폴더로 복사 + `_manifest.json` |

- 1·2단계 mouth = 비브라토 적용 **전** 원본 오프라인 궤적(`base_mouth`), 3단계 mouth = 비브라토 적용본. 헤드는 세 단계 공통.

## 출력 파일 포맷 (헤더 없음, 40ms/행 = 25fps, 세 파일 행 수 동일)

| 파일 | 열 | 비고 |
|---|---|---|
| `<곡>.csv` (head) | `roll, pitch, yaw` | |
| `<곡>-delta-big.csv` (mouth) | `mouth_tick, head_gesture_ratio, roll_comp_mouth` | 로봇은 col0·col1 만 사용 |
| `<곡>-led.csv` (led) | `brightness` 1열 (0~1) | 저장 시 지각 감마 1.8 적용 |

재생할 때는 원곡 `music/<곡>.wav` 와 0행=0초로 동시에 시작한다.

## 튜닝 위치

- 비브라토 세기: `run.py --vibrato-max-ticks 30 --vibrato-extent-reference 45 --vibrato-fade-in-s 0.12`, 세부 게이트는 `p2b_vibrato_motion.VibratoMotionConfig`
- 입 벌림/필터: `p2_offline_motion.offline_motion_generation` 상단 상수(`min_MOUTH`, `ratio_minOPEN`, `num_div`, `thresholdnum`)
- LED 연출: `p3_led_allin1.py` 상단(`LABEL_PLAN`, `WAVE_AMP`, `DIM_DUCK`, `BRIGHTNESS_GAIN`, `DROP_JUMP_MIN` …) → `run.py <곡> --led-only` 로 빠르게 반복

## 주의

- WSL 멀티라인 명령은 CRLF 로 깨지므로 `p1` 은 한 줄 `&&` 체인만 쓴다. 사내망 SSL 인터셉트 대응으로 `p1` 의 `_ENVP` 가 WSL 시스템 신뢰소를 쓰게 고정한다.
- `data/segments_offline/*.npy` 가 없으면 헤드 궤적 단계에서 실패한다.
