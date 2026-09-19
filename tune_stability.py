#!/usr/bin/env python
"""チャンク間のトーン揺れを設定別に客観指標で比較する。

参照は常に元クリップに固定する（直前チャンクを prompt に回すと
音程が暴走することが実測で判明したため、その方式は使わない）。

指標:
  ばらつき  … チャンク間 F0 の標準偏差（半音）小さいほど安定
  最大跳躍  … 隣接チャンク間の F0 差の最大（半音）
  参照ズレ  … 参照音声の F0 からの中央値の乖離（半音）0 に近いほど似ている
"""
import re
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from voxcpm import VoxCPM

ROOT = Path(__file__).parent
REF_WAV = ROOT / "work/ref/reference.wav"
REF_TXT = ROOT / "work/ref/reference.txt"
OUT = ROOT / "work/tune"
N_CHUNKS = 6
MAX_CHARS = 80

CONFIGS = {
    "A_cfg2.0_t10": dict(cfg_value=2.0, inference_timesteps=10),
    "B_cfg2.3_t20": dict(cfg_value=2.3, inference_timesteps=20),
    "C_cfg2.7_t25": dict(cfg_value=2.7, inference_timesteps=25),
    "D_cfg3.0_t30": dict(cfg_value=3.0, inference_timesteps=30),
}


def chunks_from(path: Path, n: int) -> list[str]:
    out: list[str] = []
    for para in [p.strip() for p in path.read_text(encoding="utf-8").split("\n\n") if p.strip()]:
        for part in [p for p in re.split(r"(?<=。)", para) if p.strip()]:
            if out and len(out[-1]) + len(part) <= MAX_CHARS:
                out[-1] += part
            else:
                out.append(part)
            if len(out) >= n:
                return out[:n]
    return out[:n]


def median_f0(audio: np.ndarray, rate: int) -> float:
    f0, voiced, _ = librosa.pyin(audio, sr=rate, fmin=55, fmax=400, frame_length=1024)
    vals = f0[voiced & ~np.isnan(f0)]
    return float(np.median(vals)) if vals.size else float("nan")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    texts = chunks_from(ROOT / "scripts_text/gongitsune.txt", N_CHUNKS)

    ref_audio, _ = librosa.load(str(REF_WAV), sr=16000)
    ref_f0 = median_f0(ref_audio, 16000)
    print(f"参照 F0 = {ref_f0:.1f}Hz / {len(texts)} チャンクで比較\n")

    model = VoxCPM.from_pretrained(
        str(ROOT / "VoxCPM/pretrained_models/VoxCPM2"), load_denoiser=False
    )
    rate = model.tts_model.sample_rate
    prompt_text = REF_TXT.read_text(encoding="utf-8").strip()

    rows = []
    for name, cfg in CONFIGS.items():
        f0s, pieces = [], []
        for i, chunk in enumerate(texts):
            audio = np.asarray(
                model.generate(
                    text=chunk,
                    prompt_wav_path=str(REF_WAV),   # 常に元の参照に固定
                    prompt_text=prompt_text,
                    reference_wav_path=str(REF_WAV),
                    seed=42,
                    **cfg,
                ),
                dtype=np.float32,
            )
            pieces.append(audio)
            f0s.append(median_f0(audio, rate))
            print(f"  {name} [{i+1}/{len(texts)}] F0={f0s[-1]:.1f}Hz", flush=True)

        sf.write(OUT / f"{name}.wav", np.concatenate(pieces), rate)
        arr = np.array(f0s)
        semi = 12 * np.log2(arr / np.nanmedian(arr))
        rows.append((name, np.nanstd(semi), np.nanmax(np.abs(np.diff(semi))),
                     12 * np.log2(np.nanmedian(arr) / ref_f0), arr))

    print(f"\n{'config':14s} {'ばらつき':>9s} {'最大跳躍':>9s} {'参照ズレ':>9s}  F0推移")
    for name, sd, jump, dev, arr in rows:
        traj = " ".join(f"{v:.0f}" for v in arr)
        print(f"{name:14s} {sd:7.2f}st {jump:7.2f}st {dev:+7.2f}st  {traj}")
    print(f"\n試聴: {OUT}/<config>.wav")


if __name__ == "__main__":
    main()
