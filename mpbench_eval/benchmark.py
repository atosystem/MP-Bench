"""Load the 927-task MP-Bench ground truth used in the paper."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TASKS = PACKAGE_ROOT / "benchmark" / "tasks.jsonl"

FAMILIES = ("understanding", "discussion", "turnbased")


@dataclass(frozen=True)
class Task:
    task_dir: str
    family: str
    should_speak: bool
    metatranscript: str
    question_text: str
    answer_text: str
    is_speaker_name: bool


def load_tasks(path: Path | str | None = None) -> dict[str, Task]:
    path = Path(path) if path else DEFAULT_TASKS
    tasks: dict[str, Task] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            tasks[obj["task_dir"]] = Task(
                task_dir=obj["task_dir"],
                family=obj["family"],
                should_speak=bool(obj["should_speak"]),
                metatranscript=obj.get("metatranscript") or "",
                question_text=obj.get("question_text") or "",
                answer_text=obj.get("answer_text") or "",
                is_speaker_name=bool(obj.get("is_speaker_name")),
            )
    if not tasks:
        raise ValueError(f"No tasks in {path}")
    return tasks
