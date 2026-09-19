#!/usr/bin/env python
"""長文テキストをクローン音声で朗読して 1 本の音声にまとめる。

チャンク境界で声質・音程が飛ぶのを防ぐため:
  - prompt / reference とも常に元の参照クリップに固定する
  - チャンク間の音量差を ±3dB まで補正

直前チャンクを prompt に回す continuation 方式は試して失敗した。
1 チャンクの揺れを次が引き継いで増幅し、参照 104Hz に対して
3 チャンク目以降 366Hz に固着した（幅 23 半音）。採用しないこと。

使い方:
    VoxCPM/.venv/bin/python narrate.py scripts_text/gongitsune.txt [出力名]
"""
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from voxcpm import VoxCPM

ROOT = Path(__file__).parent
REF_WAV = ROOT / "work/ref/reference.wav"
REF_TXT = ROOT / "work/ref/reference.txt"
OUT_DIR = ROOT / "outputs"

MAX_CHARS = 80       # 1 チャンク上限。丸ごと次の prompt に使うので短めに保つ
GAIN_LIMIT_DB = 3.0  # チャンク間の音量補正の上限
PAUSE_SENT = 0.25
PAUSE_PARA = 0.75
SEED = 42


def split_paragraph(paragraph: str) -> list[str]:
    if len(paragraph) <= MAX_CHARS:
        return [paragraph]
    parts = [p for p in re.split(r"(?<=。)", paragraph) if p.strip()]
    chunks: list[str] = []
    for part in parts:
        if chunks and len(chunks[-1]) + len(part) <= MAX_CHARS:
            chunks[-1] += part
        else:
            chunks.append(part)
    return chunks


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) or 1e-9


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = Path(sys.argv[1])
    name = sys.argv[2] if len(sys.argv) > 2 else src.stem

    paragraphs = [p.strip() for p in src.read_text(encoding="utf-8").split("\n\n") if p.strip()]
    plan = [split_paragraph(p) for p in paragraphs]
    total = sum(len(c) for c in plan)
    print(f"{len(paragraphs)} 段落 / {total} チャンク")

    model = VoxCPM.from_pretrained(
        str(ROOT / "VoxCPM/pretrained_models/VoxCPM2"),
        load_denoiser=False,
    )
    rate = model.tts_model.sample_rate
    base_prompt_text = REF_TXT.read_text(encoding="utf-8").strip() if REF_TXT.exists() else ""

    target_rms: float | None = None
    pieces: list[np.ndarray] = []
    done = 0

    for pi, chunks in enumerate(plan):
        if pi:
            pieces.append(np.zeros(int(rate * PAUSE_PARA), dtype=np.float32))
        for ci, chunk in enumerate(chunks):
            if ci:
                pieces.append(np.zeros(int(rate * PAUSE_SENT), dtype=np.float32))

            kwargs = dict(
                text=chunk,
                reference_wav_path=str(REF_WAV),  # 元の声にアンカーし続ける
                cfg_value=2.0,
                inference_timesteps=10,
                seed=SEED,
            )
            if base_prompt_text:
                kwargs["prompt_wav_path"] = str(REF_WAV)
                kwargs["prompt_text"] = base_prompt_text

            audio = np.asarray(model.generate(**kwargs), dtype=np.float32)

            # チャンク間の音量差だけ緩く揃える
            if target_rms is None:
                target_rms = rms(audio)
            else:
                gain = np.clip(target_rms / rms(audio),
                               10 ** (-GAIN_LIMIT_DB / 20), 10 ** (GAIN_LIMIT_DB / 20))
                audio = audio * gain
            pieces.append(audio)

            # 直前チャンクを prompt に回す方式は音程が暴走したため使わない
            # （実測: 104Hz の参照に対し 3 チャンク目以降 366Hz に固着、幅 23 半音）。
            # prompt は常に元の参照クリップに固定する。

            done += 1
            print(f"[{done}/{total}] {chunk[:28]}...", flush=True)

    wav = np.concatenate(pieces)
    peak = float(np.max(np.abs(wav)))
    if peak > 0.99:
        wav = wav / peak * 0.99

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"{name}.wav"
    sf.write(out, wav, rate)
    print(f"saved: {out}  ({len(wav) / rate / 60:.1f}分)")


if __name__ == "__main__":
    main()
