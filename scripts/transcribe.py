#!/usr/bin/env python3
"""Transcribe MP-Bench session audio with Parakeet TDT 0.6B v2.

Each scenario directory is expected to contain:

    all_stitched.wav   input audio, including the trailing silence
    response.wav       model audio on that same timeline
    combined.wav       input and model together on that timeline

JSON transcripts are written with the same relative paths and the same
Whisper-style schema the eval package reads: {"text", "chunks": [{"text",
"timestamp": [start, end]}]}.

Decoding defaults to one file at a time. A larger batch matched that on a
four-file check, but on the full GPT set it changed some transcripts, so
leave --batch_size at 1 when the stored timestamps have to be reproduced.
Split scenarios across GPUs with --shard_index and --shard_count.

Example:
    python scripts/transcribe.py \
        --input_dir /path/to/scenario_wavs \
        --output_dir /path/to/scenario_jsons
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

DEFAULT_STEMS = ("all_stitched", "response", "combined")
DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v2"


def _mono(waveform):
    if getattr(waveform, "ndim", 1) > 1:
        return waveform.mean(axis=1)
    return waveform


def prepare_wav(audio_path: Path) -> str:
    """Rewrite audio as mono PCM WAV. NeMo transcribes that temporary file."""
    waveform, sr = sf.read(audio_path)
    waveform = _mono(waveform)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    sf.write(tmp.name, waveform, sr)
    return tmp.name


def _hypotheses(outputs):
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    return list(outputs)


def _word_timestamps(hypothesis) -> list[dict]:
    stamp = getattr(hypothesis, "timestamp", None)
    if stamp is None:
        stamp = hypothesis.timestep
    return stamp["word"]


def transcript_from_hypothesis(hypothesis) -> dict:
    chunks = []
    words = []
    for word in _word_timestamps(hypothesis):
        token = word["word"]
        words.append(token)
        chunks.append({"text": token, "timestamp": [word["start"], word["end"]]})
    return {"text": " ".join(words).strip(), "chunks": chunks}


def transcribe_paths(asr_model, audio_paths: list[str], batch_size: int) -> list[dict]:
    """Transcribe prepared wav paths. batch_size 1 is one NeMo call per file."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    results = []
    step = 1 if batch_size == 1 else batch_size
    for start in range(0, len(audio_paths), step):
        chunk = audio_paths[start : start + step]
        outputs = asr_model.transcribe(
            chunk,
            timestamps=True,
            batch_size=1 if batch_size == 1 else len(chunk),
            num_workers=0,
            verbose=False,
        )
        for hypothesis in _hypotheses(outputs):
            results.append(transcript_from_hypothesis(hypothesis))
    return results


def find_audio_files(input_dir: Path, stems: set[str] | None) -> list[dict]:
    files = []
    for dirpath, _dirnames, filenames in os.walk(input_dir):
        root = Path(dirpath)
        for filename in filenames:
            path = root / filename
            if path.suffix.lower() != ".wav":
                continue
            if stems is not None and path.stem not in stems:
                continue
            relative_dir = root.relative_to(input_dir).as_posix()
            if relative_dir == ".":
                relative_dir = ""
            files.append(
                {
                    "audio_path": path,
                    "relative_dir": relative_dir,
                    "output_name": path.stem + ".json",
                    "instance": relative_dir.split("/", 1)[0] if relative_dir else "",
                }
            )
    files.sort(key=lambda item: (item["relative_dir"], item["output_name"]))
    return files


def output_path(output_dir: Path, item: dict) -> Path:
    if item["relative_dir"]:
        return output_dir / item["relative_dir"] / item["output_name"]
    return output_dir / item["output_name"]


def write_transcript(path: Path, transcript: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(transcript, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True, help="Root of scenario_*/{all_stitched,response,combined}.wav")
    parser.add_argument("--output_dir", type=Path, required=True, help="Root for the matching JSON transcripts")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"NeMo pretrained name (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--stems",
        nargs="+",
        default=list(DEFAULT_STEMS),
        help="Wav stems to transcribe (default: all_stitched response combined). Pass 'all' for every wav.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Files per NeMo call. Keep at 1 to reproduce one-file transcripts.")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Transcribe at most this many files (0 = all)")
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    args = parser.parse_args()

    if args.batch_size < 1:
        raise SystemExit("--batch_size must be >= 1")
    if args.shard_count < 1 or not (0 <= args.shard_index < args.shard_count):
        raise SystemExit("--shard_index must be in [0, --shard_count)")

    input_dir = args.input_dir.resolve()
    stems = None if args.stems == ["all"] else set(args.stems)
    files = find_audio_files(input_dir, stems)
    if args.shard_count > 1:
        instances = sorted({item["instance"] for item in files})
        keep = {name for i, name in enumerate(instances) if i % args.shard_count == args.shard_index}
        files = [item for item in files if item["instance"] in keep]
        print(f"Shard {args.shard_index}/{args.shard_count}: {len(keep)} scenario folders")
    if args.limit:
        files = files[: args.limit]

    print(f"Input: {input_dir}")
    print(f"Output: {args.output_dir.resolve()}")
    print(f"Files: {len(files)}  batch_size: {args.batch_size}")
    if args.dry_run:
        for item in files[:20]:
            rel = f"{item['relative_dir']}/{item['output_name']}" if item["relative_dir"] else item["output_name"]
            print(f"  {rel}")
        if len(files) > 20:
            print(f"  ... and {len(files) - 20} more")
        return
    if not files:
        print("No wav files matched.")
        return

    pending = []
    skipped = 0
    for item in files:
        dest = output_path(args.output_dir, item)
        if args.skip_existing and dest.is_file():
            skipped += 1
            continue
        pending.append(item)
    print(f"To transcribe: {len(pending)}  skipped existing: {skipped}")
    if not pending:
        return

    import nemo.collections.asr as nemo_asr

    print(f"Loading {args.model}")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=args.model).cuda()

    processed = 0
    errors = 0
    batch = args.batch_size
    for start in tqdm(range(0, len(pending), batch), desc="Transcribing"):
        group = pending[start : start + batch]
        temps = []
        try:
            temps = [prepare_wav(item["audio_path"]) for item in group]
            transcripts = transcribe_paths(asr_model, temps, batch_size=len(group) if batch > 1 else 1)
            if len(transcripts) != len(group):
                raise RuntimeError(f"expected {len(group)} hypotheses, got {len(transcripts)}")
            for item, transcript in zip(group, transcripts):
                write_transcript(output_path(args.output_dir, item), transcript)
                processed += 1
        except Exception as exc:
            errors += len(group)
            names = ", ".join(str(item["audio_path"]) for item in group)
            print(f"Error processing {names}: {exc}")
        finally:
            for temp in temps:
                try:
                    os.unlink(temp)
                except OSError:
                    pass

    print(f"Processed: {processed}  skipped: {skipped}  errors: {errors}")
    print(f"Transcripts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
