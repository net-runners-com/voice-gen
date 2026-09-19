#!/usr/bin/env python
"""sample.mp3 の話者をリファレンスにして音声を生成する（Ultimate Cloning）。

使い方:
    VoxCPM/.venv/bin/python generate.py "しゃべらせたいテキスト" [出力名]

参照音声を差し替えたい場合は work/ref/reference.wav と
work/ref/reference.txt（その書き起こし）を置き換える。
"""
import sys
from pathlib import Path

import soundfile as sf
from voxcpm import VoxCPM

ROOT = Path(__file__).parent
REF_WAV = ROOT / "work/ref/reference.wav"
REF_TXT = ROOT / "work/ref/reference.txt"
OUT_DIR = ROOT / "outputs"

DEFAULT_TEXT = (
    "こんにちは。こちらはリファレンス音声から生成した合成ボイスのサンプルです。"
    "文章の長さやイントネーションを確認してみてください。"
)


def main() -> None:
    text = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEXT
    name = sys.argv[2] if len(sys.argv) > 2 else "clone"

    model = VoxCPM.from_pretrained(
        str(ROOT / "VoxCPM/pretrained_models/VoxCPM2"),
        load_denoiser=False,
    )

    kwargs = dict(
        text=text,
        reference_wav_path=str(REF_WAV),
        cfg_value=2.0,
        inference_timesteps=10,
        seed=42,
    )
    # 書き起こしがあれば Ultimate Cloning（同じクリップを prompt にも渡すと再現度が上がる）
    prompt_text = REF_TXT.read_text(encoding="utf-8").strip() if REF_TXT.exists() else ""
    if prompt_text:
        kwargs["prompt_wav_path"] = str(REF_WAV)
        kwargs["prompt_text"] = prompt_text

    wav = model.generate(**kwargs)

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"{name}.wav"
    sf.write(out, wav, model.tts_model.sample_rate)
    print(f"saved: {out}  ({len(wav) / model.tts_model.sample_rate:.2f}s)")


if __name__ == "__main__":
    main()
