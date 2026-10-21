"""★ 진입점 — music/ 에 넣은 원곡 wav → allin1 분리 → 오프라인 궤적 → back 조명 → 비브라토, 단계별 저장.

  [1] allin1 분석 + demucs 분리 (WSL)      p1_allin1_separate   → 분석/<곡>/
  [2] 입/헤드 궤적 (+비브라토)              p2_offline_motion    → 궤적/<곡>/_중간산출물/
  [3] back 조명(LED) 밝기 궤적              p3_led_allin1        → 궤적/<곡>/_중간산출물/
  [4] 단계별 폴더 정리                                           → 궤적/<곡>/1_… 2_… 3_…

단계별 출력(누적):
  1_오프라인궤적/          <곡>.csv(head)  <곡>-delta-big.csv(mouth, 비브라토 X)
  2_오프라인궤적+조명/     + <곡>-led.csv
  3_전체_비브라토포함/      <곡>.csv  <곡>-delta-big.csv(비브라토 O)  <곡>-led.csv   ← Pi 배포 기본

사용:
  python run.py                       # music/ 안 전곡 (이미 완료된 곡은 건너뜀)
  python run.py Butter_BTS            # 한 곡
  python run.py Butter_BTS --force    # 궤적 다시 생성(allin1 캐시는 재사용)
  python run.py Butter_BTS --refresh  # allin1 재분석까지 전부 다시
  python run.py Butter_BTS --led-only # 모션 재사용, LED 만 다시(튜닝 반복)
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from common import (STAGES, head_csv_name, intermediate_dir, led_csv_name, list_raw_songs,
                    mouth_csv_name, resolve_raw, stage_dir, traj_dir, utf8_stdout)


def _count_rows(p: Path) -> int:
    with Path(p).open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _is_done(song: str) -> bool:
    d = stage_dir(song, 3)
    return all((d / n).exists() for n in (head_csv_name(song), mouth_csv_name(song), led_csv_name(song)))


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def process_song(raw: Path, *, force: bool = False, refresh: bool = False, led_only: bool = False,
                 vibrato_max_ticks: float = 30.0, vibrato_extent_reference: float = 45.0,
                 vibrato_fade_in_s: float = 0.12) -> dict:
    song = raw.stem
    t0 = time.time()
    print("#" * 70)
    print(f"# {song}   ({raw})")
    print("#" * 70)

    mid = intermediate_dir(song)
    mid.mkdir(parents=True, exist_ok=True)

    # ── [1] allin1 분석 + 분리 ────────────────────────────────────────
    import p1_allin1_separate as p1
    print("\n[1] allin1 구조 분석 + demucs 분리 ...")
    A = p1.analyze_and_separate(song, raw, refresh=refresh)
    print(f"  BPM {A['bpm']:.1f} | beats {len(A['beats'])} | segments {len(A['segments'])} | 출처 {A['source']}")
    print(f"  vocals: {A['vocals'].name} | MR: {A['mr'].name}")

    # ── [2] 입/헤드(+비브라토) 궤적 ───────────────────────────────────
    head_src = mid / head_csv_name(song)
    mouth_vib_src = mid / mouth_csv_name(song)
    mouth_plain_src = mid / f"{song}-delta-big_novibrato.csv"
    motion_ready = head_src.exists() and mouth_vib_src.exists() and mouth_plain_src.exists()
    if led_only and motion_ready:
        print("\n[2] 입/헤드 궤적 재사용(--led-only)")
    elif motion_ready and not force:
        print("\n[2] 입/헤드 궤적 이미 있음 → 재사용 (다시 만들려면 --force)")
    else:
        import p2_offline_motion as p2
        print("\n[2] 입/헤드(+비브라토) 궤적 생성 ...")
        p2.offline_motion_generation(
            str(A["vocals"]), run_dir=mid, song_name=song,
            vibrato_max_ticks=vibrato_max_ticks,
            vibrato_extent_reference=vibrato_extent_reference,
            vibrato_fade_in_s=vibrato_fade_in_s,
        )
    n_frames = _count_rows(mouth_vib_src)

    # ── [3] back 조명(LED) ────────────────────────────────────────────
    import p3_led_allin1 as p3
    print("\n[3] back 조명(LED) 궤적 — allin1 설계도(라벨→연출, 박자 락 둥둥) ...")
    brightness, rows, mix, bpm = p3.generate_led(song, A, raw, n_frames=n_frames)
    led_src = mid / led_csv_name(song)
    p3.write_led_csv(led_src, brightness)
    led_summary = p3.summarize(song, rows, mix, bpm)
    (mid / f"{song}-led_summary.txt").write_text(led_summary, encoding="utf-8")
    print(led_summary)

    # ── [4] 단계별 폴더 정리 ──────────────────────────────────────────
    print("\n[4] 단계별 저장 ...")
    for k in (1, 2, 3):
        d = stage_dir(song, k)
        if d.exists():
            shutil.rmtree(d)
    # 1: 오프라인 궤적만 (조명 X, 비브라토 X)
    _copy(head_src, stage_dir(song, 1) / head_csv_name(song))
    _copy(mouth_plain_src, stage_dir(song, 1) / mouth_csv_name(song))
    # 2: + back 조명
    _copy(head_src, stage_dir(song, 2) / head_csv_name(song))
    _copy(mouth_plain_src, stage_dir(song, 2) / mouth_csv_name(song))
    _copy(led_src, stage_dir(song, 2) / led_csv_name(song))
    # 3: + 비브라토 (전체)
    _copy(head_src, stage_dir(song, 3) / head_csv_name(song))
    _copy(mouth_vib_src, stage_dir(song, 3) / mouth_csv_name(song))
    _copy(led_src, stage_dir(song, 3) / led_csv_name(song))

    rows_head, rows_led = _count_rows(head_src), _count_rows(led_src)
    manifest = {
        "song": song,
        "raw_wav": str(raw),
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "allin1_source": A["source"],
        "bpm": A["bpm"],
        "n_frames": n_frames,
        "frame_ms": 40,
        "rows": {"head": rows_head, "mouth": n_frames, "led": rows_led},
        "vibrato": {"max_ticks": vibrato_max_ticks, "extent_reference_cents": vibrato_extent_reference,
                    "fade_in_s": vibrato_fade_in_s},
        "stages": {
            STAGES[1]: {"head": head_csv_name(song), "mouth": mouth_csv_name(song), "led": None,
                        "note": "오프라인 궤적만 — 조명 X, 비브라토 X"},
            STAGES[2]: {"head": head_csv_name(song), "mouth": mouth_csv_name(song), "led": led_csv_name(song),
                        "note": "오프라인 궤적 + back 조명 — 비브라토 X"},
            STAGES[3]: {"head": head_csv_name(song), "mouth": mouth_csv_name(song), "led": led_csv_name(song),
                        "note": "전체 — 비브라토 포함(Pi 배포 기본)"},
        },
    }
    (traj_dir(song) / "_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
    for k in (1, 2, 3):
        print(f"  {STAGES[k]:22s} ← {', '.join(p.name for p in sorted(stage_dir(song, k).iterdir()))}")
    if not (rows_head == n_frames == rows_led):
        print(f"  ⚠ 행 수 불일치: head {rows_head} / mouth {n_frames} / led {rows_led}")
    else:
        print(f"  행 수 {n_frames} (40ms × {n_frames} ≈ {n_frames * 0.04:.1f}s) — head/mouth/led 정렬 OK")
    print(f"\n[완료] {song}  총 {time.time() - t0:.1f}s  → {traj_dir(song)}")
    return manifest


def main():
    utf8_stdout()
    ap = argparse.ArgumentParser(description="원곡 wav → allin1 분리 → 오프라인 궤적/조명/비브라토 단계별 생성")
    ap.add_argument("songs", nargs="*", help="곡 이름(들). 비우면 music/ 전곡")
    ap.add_argument("--force", action="store_true", help="궤적 다시 생성(allin1 캐시는 재사용)")
    ap.add_argument("--refresh", action="store_true", help="allin1 재분석까지 전부 다시")
    ap.add_argument("--led-only", action="store_true", help="모션 재사용, LED 만 다시")
    ap.add_argument("--vibrato-max-ticks", type=float, default=30.0)
    ap.add_argument("--vibrato-extent-reference", type=float, default=45.0)
    ap.add_argument("--vibrato-fade-in-s", type=float, default=0.12)
    args = ap.parse_args()

    if args.songs:
        raws = [resolve_raw(s) for s in args.songs]
    else:
        raws = list_raw_songs()
        if not raws:
            print("music/ 에 원곡 wav 가 없습니다. music/<곡>.wav 를 넣고 다시 실행하세요.")
            return
        skip = [r for r in raws if _is_done(r.stem) and not (args.force or args.refresh or args.led_only)]
        if skip:
            print(f"이미 완료된 곡 {len(skip)}개 건너뜀 (다시 하려면 --force): "
                  + ", ".join(r.stem for r in skip))
        raws = [r for r in raws if r not in skip]
        if not raws:
            print("처리할 곡이 없습니다.")
            return

    print(f"처리 대상 {len(raws)}곡: " + ", ".join(r.stem for r in raws))
    t_all = time.time()
    failed = []
    for raw in raws:
        try:
            process_song(raw, force=args.force or args.refresh, refresh=args.refresh, led_only=args.led_only,
                         vibrato_max_ticks=args.vibrato_max_ticks,
                         vibrato_extent_reference=args.vibrato_extent_reference,
                         vibrato_fade_in_s=args.vibrato_fade_in_s)
        except Exception as e:   # 한 곡 실패가 배치 전체를 멈추지 않도록
            failed.append((raw.stem, repr(e)))
            print(f"\n[실패] {raw.stem}: {e!r}")
            if len(raws) == 1:
                raise
        print()

    print("=" * 70)
    print(f"전체 {len(raws)}곡 중 성공 {len(raws) - len(failed)} / 실패 {len(failed)}  (총 {time.time() - t_all:.0f}s)")
    for s, e in failed:
        print(f"  ✗ {s}: {e}")


if __name__ == "__main__":
    main()
