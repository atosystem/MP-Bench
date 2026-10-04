"""Paper Table 3 metrics."""

from __future__ import annotations

from statistics import mean

from .benchmark import Task
from .transcripts import Detected


def _rate(k: int, n: int) -> dict:
    if n == 0:
        return {"count": 0, "total": 0, "pct": None}
    return {"count": k, "total": n, "pct": round(100.0 * k / n, 2)}


def _lat(values: list[float]) -> float | None:
    return None if not values else round(mean(values), 1)


def _verdict(judgments: dict[str, bool], kind: str, task_dir: str) -> bool | None:
    return judgments.get(f"{kind}:{task_dir}")


def score(detected: list[Detected], tasks: dict[str, Task], judgments: dict[str, bool]) -> dict:
    """Nine Table 3 numbers.

    * Turn-taking: spoke after the last input word vs. `should_speak`.
    * Understanding Overall / Spk. Name: QA judge and speaker-name fuzzy judge.
    * App Acc: consistency judge. Empty post-input speech is inappropriate,
      except a correct silence on a Turn-based negative turn.
    * Latency: mean first-word gap over responses that spoke after input.
    """
    by = {name: [r for r in detected if r.family == name] for name in ("understanding", "discussion", "turnbased")}

    def yes(kind, row, empty_is=False, silent_negative_ok=False):
        verdict = _verdict(judgments, kind, row.task_dir)
        if verdict is not None:
            return verdict
        if silent_negative_ok and (not row.spoke) and (not row.should_speak):
            return True
        if not row.response.strip():
            return empty_is
        return False

    und, disc, tb = by["understanding"], by["discussion"], by["turnbased"]
    spk = [r for r in und if tasks[r.task_dir].is_speaker_name]

    return {
        "understanding": {
            "n": len(und),
            "overall_acc": _rate(sum(yes("understanding", r) for r in und), len(und)),
            "spk_name_acc": _rate(sum(yes("speaker_name", r) for r in spk), len(spk)),
            "latency_s": _lat([r.latency_s for r in und if r.spoke and r.latency_s is not None]),
        },
        "discussion": {
            "n": len(disc),
            "tt_acc": _rate(sum(r.tt_correct for r in disc), len(disc)),
            "app_acc": _rate(sum(yes("discussion", r) for r in disc), len(disc)),
            "latency_s": _lat([r.latency_s for r in disc if r.spoke and r.latency_s is not None]),
        },
        "turnbased": {
            "n": len(tb),
            "tt_acc": _rate(sum(r.tt_correct for r in tb), len(tb)),
            "app_acc": _rate(sum(yes("turnbased", r, silent_negative_ok=True) for r in tb), len(tb)),
            "latency_s": _lat([r.latency_s for r in tb if r.spoke and r.latency_s is not None]),
        },
    }


def render_markdown(model: str, result: dict) -> str:
    def pct(stat):
        if not stat or stat["pct"] is None:
            return "—"
        return f"{stat['pct']:.2f}% ({stat['count']}/{stat['total']})"

    def lat(value):
        return "—" if value is None else f"{value:.1f}"

    u, d, t = result["understanding"], result["discussion"], result["turnbased"]
    return "\n".join(
        [
            f"# MP-Bench results — {model}",
            "",
            "| Split | Overall Acc. | Spk. Name Acc. | Lat. (s) | TT Acc. | App Acc. | Lat. (s) |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| Understanding: Discussion | {pct(u['overall_acc'])} | {pct(u['spk_name_acc'])} | {lat(u['latency_s'])} |  |  |  |",
            f"| Behavior: Discussion |  |  |  | {pct(d['tt_acc'])} | {pct(d['app_acc'])} | {lat(d['latency_s'])} |",
            f"| Behavior: Turn-based |  |  |  | {pct(t['tt_acc'])} | {pct(t['app_acc'])} | {lat(t['latency_s'])} |",
            "",
        ]
    )
