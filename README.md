# voice-gen

`sample.mp3`（本人の録音）の声を [VoxCPM2](https://github.com/OpenBMB/VoxCPM) でクローンして、任意のテキストや長文を読み上げるスクリプト集。

## 構成

| パス | 役割 |
|---|---|
| `sample.mp3` | クローン元の録音（BGM 入り） |
| `work/ref/reference.wav` / `.txt` | `sample.mp3` から BGM を除いて切り出した参照クリップと書き起こし |
| `prepare_reference.py` | 参照クリップを作り直す（demucs でボーカル分離 → 16kHz モノラル化） |
| `generate.py` | 短いテキストを 1 本生成 |
| `narrate.py` | 長文をチャンク分割して朗読し、1 本にまとめる |
| `tune_stability.py` | チャンク間の音程の揺れを設定別に比較する |
| `gemini_tts.py` | Gemini TTS で朗読（プリセット声のみ・クローン不可） |
| `scripts_text/` | 朗読用テキスト |

## セットアップ

```bash
git clone https://github.com/OpenBMB/VoxCPM.git
cd VoxCPM
uv venv --python 3.12
uv pip install -e . demucs librosa
.venv/bin/hf download openbmb/VoxCPM2 --local-dir pretrained_models/VoxCPM2
cd ..
```

`ffmpeg` も必要（`brew install ffmpeg`）。Apple Silicon では MPS で動く。

## 使い方

```bash
# 短文
VoxCPM/.venv/bin/python generate.py "しゃべらせたいテキスト" 出力名

# 長文朗読
VoxCPM/.venv/bin/python narrate.py scripts_text/gongitsune.txt gongitsune

# 参照クリップを別区間で作り直す（例: 7:32 から 11.65 秒）
VoxCPM/.venv/bin/python prepare_reference.py 452.22 11.65 reference

# Gemini TTS（GEMINI_API_KEY を環境変数か .env に設定）
python gemini_tts.py scripts_text/gongitsune.txt Charon
```

出力は `outputs/` に保存される（git 管理外）。

## メモ

- 参照音声に BGM が残っていると音楽ごとクローンされる。必ず `prepare_reference.py` で分離したものを使う。
- 長文で直前チャンクを次の prompt に回す方式は、音程が暴走した（参照 104Hz → 366Hz）ため使わない。参照は常に元クリップに固定する。
