#!/usr/bin/env python
"""Gemini API で朗読音声を生成する。

VoxCPM と違い長文を 1 回のリクエストで生成できるので、
チャンク分割に起因するトーンのブレが原理的に起きない。
ただし声はプリセットのみで、参照音声からのクローンは API 非対応。

使い方:
    python gemini_tts.py <テキストファイル> [声名] [出力名]
    python gemini_tts.py --voices          # 低音系の声を試聴用に一括生成
"""
import base64, json, os, sys, urllib.error, urllib.request, wave
from pathlib import Path

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "outputs"
MODEL = "gemini-3.1-flash-tts-preview"
RATE = 24000

STYLE = ("落ち着いた低い声で、終始一定のトーンを保ち、"
         "感情を抑えて淡々と、ゆっくり朗読してください。\n\n")

LOW_VOICES = ["Charon", "Algenib", "Alnilam", "Gacrux", "Schedar", "Rasalgethi"]


def api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        key = (ROOT / ".env").read_text().split("=", 1)[1].strip()
    return key


def synth(text: str, voice: str) -> bytes:
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{MODEL}:generateContent?key={api_key()}")
    body = {
        "contents": [{"parts": [{"text": STYLE + text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "languageCode": "ja-JP",
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}},
            },
        },
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        res = json.load(urllib.request.urlopen(req, timeout=600))
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code}: {e.read().decode()[:500]}")
    cand = res["candidates"][0]
    parts = cand.get("content", {}).get("parts")
    if not parts:
        raise RuntimeError(f"no audio (finishReason={cand.get('finishReason')})")
    return base64.b64decode(parts[0]["inlineData"]["data"])


def save(pcm: bytes, path: Path) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
        w.writeframes(pcm)
    return len(pcm) / 2 / RATE


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--voices":
        sample = "これは、私が小さいときに、村のもへいというおじいさんからきいたお話です。"
        for v in LOW_VOICES:
            try:
                d = save(synth(sample, v), ROOT / f"work/voices/{v}.wav")
                print(f"{v:12s} {d:.2f}s")
            except RuntimeError as e:
                print(f"{v:12s} 失敗: {e}")
        return

    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = Path(sys.argv[1])
    voice = sys.argv[2] if len(sys.argv) > 2 else "Charon"
    name = sys.argv[3] if len(sys.argv) > 3 else f"{src.stem}_gemini"
    text = src.read_text(encoding="utf-8")

    out = OUT_DIR / f"{name}.wav"
    dur = save(synth(text, voice), out)
    print(f"saved: {out}  ({dur/60:.1f}分, voice={voice})")


if __name__ == "__main__":
    main()
