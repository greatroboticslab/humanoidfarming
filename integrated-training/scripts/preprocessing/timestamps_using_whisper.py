#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path
from faster_whisper import WhisperModel

def write_json(segments, lang, json_path: Path):
    seg_objs = []
    for i, (sid, start, end, text) in enumerate(segments):
        seg_objs.append({"id": i, "start": float(start), "end": float(end), "text": text.strip()})
    obj = {"text": " ".join(s["text"] for s in seg_objs), "segments": seg_objs, "language": lang or "unknown"}
    json_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def transcribe_file(model, in_fp: Path, out_dir: Path, overwrite=False, beam_size=5, vad=True, lang=None, word_ts=False):
    json_path = out_dir / f"{in_fp.stem}.json"
    if not overwrite and json_path.exists():
        print(f"✔️  Skipping (exists): {json_path.name}")
        return
    print(f"→ Transcribing: {in_fp}")
    segments, info = model.transcribe(
        str(in_fp),
        beam_size=beam_size,
        vad_filter=vad,
        language=lang,
        word_timestamps=word_ts
    )
    collected = [(seg.id, seg.start, seg.end, seg.text) for seg in segments]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(collected, info.language, json_path)
    print(f"✅ wrote {json_path.name}")

def main():
    ap = argparse.ArgumentParser(description="Whisper (faster-whisper) → JSON with segment timestamps")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--compute_type", default="float16")   # good for V100
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument("--no_vad", action="store_true")
    ap.add_argument("--language", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--word_timestamps", action="store_true")
    args = ap.parse_args()

    model = WhisperModel(
        args.model, device=args.device, compute_type=args.compute_type,
        download_root=os.environ.get("WHISPER_CACHE")  # optional cache dir
    )
    in_dir, out_dir = Path(args.input_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4s = sorted(in_dir.glob("*.mp4"))
    if not mp4s:
        print(f"No .mp4 files in {in_dir}"); return
    for fp in mp4s:
        try:
            transcribe_file(
                model, fp, out_dir,
                overwrite=args.overwrite,
                beam_size=args.beam_size,
                vad=(not args.no_vad),
                lang=args.language,
                word_ts=args.word_timestamps
            )
        except Exception as e:
            print(f"❌ Failed on {fp.name}: {e}")
if __name__ == "__main__":
    main()
