#!/usr/bin/env python
"""sample.mp3 から BGM を除いた参照音声を切り出す。

demucs でボーカル分離してから指定区間を 16kHz モノラルに整える。
BGM が乗ったままだと VoxCPM が音楽ごと声質をクローンしてしまうため必須。

使い方:
    VoxCPM/.venv/bin/python prepare_reference.py <開始秒> <長さ秒> [出力名]

例（既定の参照 = sample.mp3 の 7:32 から 11.65 秒）:
    VoxCPM/.venv/bin/python prepare_reference.py 452.22 11.65 reference
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
PY = ROOT / "VoxCPM/.venv/bin/python"
SRC = ROOT / "sample.mp3"
WORK = ROOT / "work"
REF_DIR = WORK / "ref"
PAD = 4.0  # 分離精度を上げるため前後に余白を取る


def run(*cmd) -> None:
    subprocess.run([str(c) for c in cmd], check=True)


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    start, dur = float(sys.argv[1]), float(sys.argv[2])
    name = sys.argv[3] if len(sys.argv) > 3 else "reference"

    REF_DIR.mkdir(parents=True, exist_ok=True)
    region = WORK / f"region_{name}.wav"

    # 1. 余白つきで元の品質のまま切り出す
    run("ffmpeg", "-y", "-v", "error", "-i", SRC,
        "-ss", max(0.0, start - PAD), "-t", dur + PAD * 2,
        "-ar", 44100, "-ac", 2, region)

    # 2. ボーカル分離（初回は htdemucs の重みを DL）
    run(PY, "-m", "demucs", "--two-stems=vocals", "-d", "cpu",
        "-o", WORK / "sep", region)

    # 3. 目的の区間だけ切り直してラウドネスを揃える
    vocals = WORK / "sep/htdemucs" / region.stem / "vocals.wav"
    out = REF_DIR / f"{name}.wav"
    run("ffmpeg", "-y", "-v", "error", "-i", vocals,
        "-ss", min(PAD, start), "-t", dur,
        "-af", "loudnorm=I=-20:TP=-2:LRA=7",
        "-ar", 16000, "-ac", 1, out)

    print(f"saved: {out}")
    print(f"次: {REF_DIR / (name + '.txt')} にこの区間の書き起こしを入れる")


if __name__ == "__main__":
    main()
