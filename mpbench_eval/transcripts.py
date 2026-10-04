"""Read ASR transcripts, decide if the model took the floor, and extract text/latency."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

# Spoke / judged text: model-only transcripts. `combined.json` mixes input
# and output on one timeline and is reserved for latency.
_SPEAK_LAYOUTS = (
    ("all_stitched.json", "response.json"),
    ("all_stitched/input.json", "all_stitched/output.json"),
    ("all_stitched/input.json", "response.json"),
    ("input.json", "output.json"),
    ("combined.json", "output.json"),
)
_LATENCY_LAYOUTS = (
    ("all_stitched.json", "combined.json"),
    ("all_stitched/input.json", "all_stitched/combined.json"),
    ("input.json", "combined.json"),
)


def _chunks(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [c for c in data.get("chunks", []) if c.get("timestamp") and len(c["timestamp"]) == 2]


def _full_text(path: Path) -> str:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    text = (data.get("text") or "").strip()
    if text:
        return text
    return " ".join(c.get("text", "") for c in data.get("chunks", [])).strip()


def _first_layout(scenario_dir: Path, layouts) -> tuple[Path, Path] | None:
    for inp, out in layouts:
        ip, op = scenario_dir / inp, scenario_dir / out
        if ip.is_file() and op.is_file() and ip.stat().st_size > 0 and op.stat().st_size > 0:
            return ip, op
    return None


def resolve_layout(scenario_dir: Path) -> tuple[Path, Path] | None:
    return _first_layout(scenario_dir, _SPEAK_LAYOUTS)


@dataclass
class Detected:
    task_dir: str
    family: str
    should_speak: bool
    spoke: bool
    tt_correct: bool
    latency_s: float | None
    response: str
    source: str
    # Discussion App Acc. judges post-input words only (`response`); the other
    # three judges see the whole transcribed reply.
    full_response: str = ""

    def to_json(self) -> dict:
        return asdict(self)


def _latency_from_combined(scenario_dir: Path) -> float | None:
    layout = _first_layout(scenario_dir, _LATENCY_LAYOUTS)
    if layout is None:
        return None
    in_chunks, out_chunks = _chunks(layout[0]), _chunks(layout[1])
    if not in_chunks or not out_chunks:
        return None
    input_end = max(c["timestamp"][1] for c in in_chunks)
    post = [c for c in out_chunks if c["timestamp"][0] >= input_end]
    if not post:
        return None
    return float(post[0]["timestamp"][0] - input_end)


def detect_one(scenario_dir: Path, half_duplex: bool = False) -> tuple[bool, float | None, str, str, str]:
    """Return (spoke, latency_s, post_input_text, source, full_text).

    Streaming / same-timeline dumps: spoke iff any model word starts at or
    after the last input word (paper Sec. 3.3.1). Half-duplex dumps timestamp
    the response from 0, so a nonempty transcript counts as spoke.
    """
    layout = resolve_layout(scenario_dir)
    if layout is None:
        return False, None, "", "missing", ""

    input_path, output_path = layout
    in_chunks = _chunks(input_path)
    out_chunks = _chunks(output_path)
    full = _full_text(output_path)
    latency = _latency_from_combined(scenario_dir)
    source = f"timestamps:{input_path.name}+{output_path.name}"

    if in_chunks and out_chunks:
        input_end = max(c["timestamp"][1] for c in in_chunks)
        post = [c for c in out_chunks if c["timestamp"][0] >= input_end]
        if post:
            text = " ".join(c.get("text", "") for c in post).strip()
            if latency is None:
                latency = float(post[0]["timestamp"][0] - input_end)
            return True, latency, text, source, full
        if half_duplex and full:
            if latency is None and out_chunks:
                latency = float(out_chunks[0]["timestamp"][0])
            return True, latency, full, f"halfduplex:{output_path.name}", full
        return False, None, "", source, full

    if full:
        return True, latency, full, f"text:{output_path.name}", full
    return False, None, "", f"text:{output_path.name}", full


def load_spoke_tsv(path: Path) -> dict[str, bool]:
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            value = (row.get("spoke_after_input") or "").strip().lower()
            if row.get("task_dir") and value in ("true", "false"):
                out[row["task_dir"]] = value == "true"
    return out


def detect_all(result_dir: Path, tasks: dict, spoke_tsv: Path | None = None) -> list[Detected]:
    # Probe whether this dump uses a shared timeline. If almost no task has a
    # post-input word but most have response text, treat the run as half-duplex.
    probes = []
    for task_dir in tasks:
        layout = resolve_layout(result_dir / task_dir)
        if layout is None:
            continue
        in_chunks, out_chunks = _chunks(layout[0]), _chunks(layout[1])
        full = _full_text(layout[1])
        if not in_chunks or not out_chunks:
            continue
        input_end = max(c["timestamp"][1] for c in in_chunks)
        post = any(c["timestamp"][0] >= input_end for c in out_chunks)
        probes.append((post, bool(full)))
    half_duplex = False
    if probes:
        n_post = sum(p for p, _ in probes)
        n_text = sum(t for _, t in probes)
        half_duplex = n_text > 0 and (n_post / n_text) < 0.25

    tsv = load_spoke_tsv(spoke_tsv) if spoke_tsv else {}
    rows = []
    for task_dir, task in tasks.items():
        spoke, latency, text, source, full = detect_one(result_dir / task_dir, half_duplex=half_duplex)
        if task_dir in tsv:
            spoke = tsv[task_dir]
            source = f"tsv+{source}"
        rows.append(
            Detected(
                task_dir=task_dir,
                family=task.family,
                should_speak=task.should_speak,
                spoke=spoke,
                tt_correct=(spoke == task.should_speak),
                latency_s=None if latency is None else round(latency, 3),
                response=text,
                source=source,
                full_response=full,
            )
        )
    rows.sort(key=lambda r: r.task_dir)
    return rows
