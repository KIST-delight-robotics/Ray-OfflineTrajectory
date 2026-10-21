#!/usr/bin/env bash
# allin1 GPU 환경 설치 → /root/allin1env
#   이미 설치된 WSL Ubuntu-22.04 안에서 root 로 실행한다. 보통은 setup_allin1env.bat 가 호출.
#   내용: torch 2.2.0+cu121, natten 0.15.1 (torch220cu121, GitHub 릴리스 휠), madmom(git), allin1
#
#   ※ 버전 조합: allin1 1.1.0 은 natten 0.14 시절 API(natten1dqkrpb 등)를 import 하므로 natten 은 0.15.1 까지만 호환된다
#     (0.17.x 는 ImportError). 0.15.1 의 CUDA 휠이 있는 가장 새 torch 가 2.2 → torch 2.2.0 으로 고정.
#     대안(allin1 문서의 조합): torch 2.0.0+cu118 + natten 0.14.6+torch200cu118.
#   ※ natten 휠 서버(shi-labs.com)는 사내망에서 안 열려 GitHub 릴리스의 같은 휠을 직접 받는다 (cp310 = Ubuntu 22.04 python3.10).
#   ※ SETUP_DIR: 이 스크립트가 있는 폴더의 WSL 경로(/mnt/d/.../wsl_setup). bat 가 넘겨주고, 직접 실행하면 $0 기준.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
SETUP_DIR="${SETUP_DIR:-$(cd "$(dirname "$0")" && pwd)}"
VENV=/root/allin1env
PY=$VENV/bin/python
TORCH_V=2.2.0
CUDA_TAG=cu121
NATTEN_V=0.15.1
NATTEN_WHL="https://github.com/SHI-Labs/NATTEN/releases/download/v${NATTEN_V}/natten-${NATTEN_V}%2Btorch220${CUDA_TAG}-cp310-cp310-linux_x86_64.whl"
# 사내망 SSL 인터셉트 장비의 루트 인증서. 파일이 있을 때만 등록한다(다른 망에서는 무해).
CA_SRC="$SETUP_DIR/eprism_ssl_root.crt"
CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# SETUP_DRY_RUN=1 이면 경로/버전만 보여주고 끝낸다 (bat 의 경로 처리 점검용)
if [ "${SETUP_DRY_RUN:-0}" = "1" ]; then
  echo "DRY RUN"
  echo "  SETUP_DIR  = $SETUP_DIR"
  echo "  CA_SRC     = $CA_SRC ($( [ -f "$CA_SRC" ] && echo 있음 || echo 없음 ))"
  echo "  torch      = $TORCH_V+$CUDA_TAG"
  echo "  NATTEN_WHL = $NATTEN_WHL"
  echo "SETUP_DONE"
  exit 0
fi

echo "[1/7] apt 패키지 (git, 빌드도구, ffmpeg, python3.10-venv) ..."
apt-get update -qq
apt-get install -y -qq git build-essential ffmpeg python3.10-venv python3.10-dev ca-certificates curl > /tmp/apt.log 2>&1 \
  || { tail -20 /tmp/apt.log; exit 1; }

echo "[2/7] 사내망 루트 인증서 등록 (있으면) ..."
if [ -f "$CA_SRC" ]; then
  tr -d '\r' < "$CA_SRC" > /usr/local/share/ca-certificates/eprism-ssl.crt
  update-ca-certificates > /tmp/ca.log 2>&1 || { cat /tmp/ca.log; exit 1; }
  # pip / 파이썬(requests, urllib) / curl 이 시스템 신뢰소를 쓰게 고정 (login shell 용 + 이 스크립트 용)
  printf 'export PIP_CERT=%s SSL_CERT_FILE=%s REQUESTS_CA_BUNDLE=%s CURL_CA_BUNDLE=%s\n' \
    "$CA_BUNDLE" "$CA_BUNDLE" "$CA_BUNDLE" "$CA_BUNDLE" > /etc/profile.d/zz-corp-ca.sh
  printf '[global]\ncert = %s\n' "$CA_BUNDLE" > /etc/pip.conf
  # shellcheck disable=SC1091
  . /etc/profile.d/zz-corp-ca.sh
  echo "  등록: $(openssl x509 -in "$CA_SRC" -noout -subject | sed 's/subject=//')"
else
  echo "  $CA_SRC 없음 → 건너뜀"
fi

echo "[3/7] venv $VENV ..."
test -d "$VENV" || python3.10 -m venv "$VENV"
$PY -m pip install -q --upgrade pip setuptools wheel "numpy<2" cython

echo "[4/7] torch==$TORCH_V+$CUDA_TAG + torchaudio ... (약 2.5GB 다운로드)"
$PY -m pip install -q "torch==$TORCH_V" "torchaudio==$TORCH_V" --index-url "https://download.pytorch.org/whl/$CUDA_TAG"

echo "[5/7] natten==$NATTEN_V (GitHub 릴리스 휠) ..."
$PY -m pip install -q "$NATTEN_WHL"

echo "[6/7] madmom(git) + allin1 ..."
$PY -m pip install -q git+https://github.com/CPJKU/madmom
$PY -m pip install -q allin1

echo "[7/7] 확인 ..."
$PY - <<'PYEOF'
import torch, natten, allin1
from natten.functional import natten1dqkrpb, natten1dav, natten2dqkrpb, natten2dav  # allin1 이 쓰는 옛 API
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU 없음")
print("natten", getattr(natten, "__version__", "?"), "| allin1 import OK")
PYEOF
echo "SETUP_DONE"
