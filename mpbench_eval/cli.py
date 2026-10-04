"""CLI: python -m mpbench_eval {detect,judge,report,run}"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import benchmark, judge as judge_mod, metrics
from .transcripts import Detected, detect_all

DETECT_FILE = "detect.jsonl"
JUDGE_FILE = "judgments.jsonl"
REPORT_JSON = "RESULTS.json"
REPORT_MD = "RESULTS.md"


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _out_dir(args) -> Path:
    path = Path(args.out_dir) if args.out_dir else Path(args.result_dir) / "mpbench_eval_out"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _model_name(args) -> str:
    return args.name or Path(args.result_dir).name


def cmd_detect(args) -> list[Detected]:
    tasks = benchmark.load_tasks(args.tasks)
    rows = detect_all(
        Path(args.result_dir),
        tasks,
        spoke_tsv=Path(args.spoke_tsv) if getattr(args, "spoke_tsv", None) else None,
    )
    out = _out_dir(args)
    _write_jsonl(out / DETECT_FILE, (r.to_json() for r in rows))
    by = {}
    for r in rows:
        by.setdefault(r.family, {"n": 0, "spoke": 0, "tt": 0})
        by[r.family]["n"] += 1
        by[r.family]["spoke"] += int(r.spoke)
        by[r.family]["tt"] += int(r.tt_correct)
    summary = {"model": _model_name(args), "n": len(rows), "by_family": by}
    (out / "detect_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    n_missing = sum(r.source == "missing" for r in rows)
    if n_missing:
        print(
            f"warning: {n_missing}/{len(rows)} tasks have no ASR folder under {args.result_dir}. "
            "Point --result-dir at the full 927-scenario dump, not a Discussion-only rerun.",
            file=sys.stderr,
        )
    return rows


# Only Discussion App Acc. scores post-input speech; the other judges read the
# whole reply, matching the CSVs the paper's numbers were computed from.
JUDGE_TEXT = {
    "understanding": "full_response",
    "speaker_name": "full_response",
    "turnbased": "full_response",
    "discussion": "response",
}


def _pending_items(kind: str, detected: list[dict], tasks, done: set) -> list[dict]:
    items = []
    field = JUDGE_TEXT[kind]
    for row in detected:
        task = tasks[row["task_dir"]]
        if kind == "understanding" and task.family != "understanding":
            continue
        if kind == "speaker_name" and not task.is_speaker_name:
            continue
        if kind == "discussion" and task.family != "discussion":
            continue
        if kind == "turnbased" and task.family != "turnbased":
            continue
        if (kind, row["task_dir"]) in done:
            continue
        items.append({
            "task_dir": row["task_dir"],
            "family": task.family,
            "response": row.get(field) or "",
            "spoke": row["spoke"],
            "should_speak": row["should_speak"],
        })
    return items


def cmd_judge(args) -> list[dict]:
    tasks = benchmark.load_tasks(args.tasks)
    out = _out_dir(args)
    detected = _read_jsonl(out / DETECT_FILE)
    if not detected:
        raise SystemExit(f"{out / DETECT_FILE} missing. Run `detect` first.")

    kinds = args.kinds or ["understanding", "speaker_name", "discussion", "turnbased"]
    existing = _read_jsonl(out / JUDGE_FILE) if args.resume else []
    done = {(r["kind"], r["task_dir"]) for r in existing if r.get("verdict") is not None}
    merged = {(r["kind"], r["task_dir"]): r for r in existing}

    if args.dry_run:
        preview = out / "dry_run_prompts"
        preview.mkdir(exist_ok=True)
        n = 0
        for kind in kinds:
            for item in _pending_items(kind, detected, tasks, done)[:2]:
                (preview / f"{kind}_{item['task_dir']}.txt").write_text(
                    judge_mod.build_prompt(kind, tasks[item["task_dir"]], item["response"])
                )
                n += 1
        print(f"Dry run: wrote {n} prompt previews to {preview}")
        return existing

    api_key = judge_mod.resolve_api_key(args.provider, args.api_key)
    for kind in kinds:
        items = _pending_items(kind, detected, tasks, done)
        auto = []
        api_items = []
        for item in items:
            # The turn-based CoT prompt reasons about silence itself, so empty
            # replies still go to the judge.
            if not item["response"].strip() and kind != "turnbased":
                auto.append({**item, "kind": kind, "verdict": False, "error": None, "reasoning": "empty response"})
            else:
                api_items.append(item)
        if args.limit:
            api_items = api_items[: args.limit]

        print(f"{kind}: {len(api_items)} to judge, {len(auto)} filled without API")
        progress = {"n": 0}

        def report(record, total=len(api_items)):
            progress["n"] += 1
            if record.get("error") and progress["n"] == 1:
                print(f"  {kind} judge error: {record['error'][:300]}", flush=True)
            if progress["n"] % 25 == 0 or progress["n"] == total:
                print(f"  {kind} {progress['n']}/{total}", flush=True)

        fresh = []
        if api_items:
            fresh = judge_mod.run_judge(
                kind,
                api_items,
                tasks,
                provider=args.provider,
                model=args.model,
                api_key=api_key,
                workers=args.workers,
                temperature=None if args.no_sampling_params else args.temperature,
                max_tokens=args.max_tokens,
                on_result=report,
            )
            n_err = sum(1 for r in fresh if r.get("error"))
            if n_err:
                print(f"  {kind}: {n_err}/{len(fresh)} judge calls failed", flush=True)
            if n_err == len(fresh):
                raise SystemExit(f"{kind}: every judge call failed.")
        for row in auto + fresh:
            merged[(row["kind"], row["task_dir"])] = row

    rows = [merged[k] for k in sorted(merged)]
    _write_jsonl(out / JUDGE_FILE, rows)
    print(f"Wrote {out / JUDGE_FILE} ({len(rows)} rows)")
    return rows


def _judgments_map(rows: list[dict]) -> dict[str, bool]:
    out = {}
    for row in rows:
        if isinstance(row.get("verdict"), bool):
            out[f"{row['kind']}:{row['task_dir']}"] = row["verdict"]
    return out


def cmd_report(args) -> dict:
    tasks = benchmark.load_tasks(args.tasks)
    out = _out_dir(args)
    raw = _read_jsonl(out / DETECT_FILE)
    if not raw:
        raise SystemExit(f"{out / DETECT_FILE} missing. Run `detect` first.")
    fields = Detected.__dataclass_fields__
    detected = [Detected(**{k: r[k] for k in fields if k in r}) for r in raw]
    judgments = _judgments_map(_read_jsonl(out / JUDGE_FILE))
    result = metrics.score(detected, tasks, judgments)
    payload = {"model": _model_name(args), "result_dir": str(args.result_dir), **result}
    (out / REPORT_JSON).write_text(json.dumps(payload, indent=2) + "\n")
    md = metrics.render_markdown(payload["model"], result)
    (out / REPORT_MD).write_text(md)
    print(md)
    print(f"Wrote {out / REPORT_JSON}")
    return payload


def cmd_run(args):
    cmd_detect(args)
    cmd_judge(args)
    return cmd_report(args)


def _add_common(p):
    p.add_argument("--result-dir", required=True, help="Directory of scenario_* ASR transcripts")
    p.add_argument("--out-dir", default=None, help="Default: <result-dir>/mpbench_eval_out")
    p.add_argument("--name", default=None)
    p.add_argument("--tasks", default=None, help="Override benchmark/tasks.jsonl")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mpbench_eval",
        description="Evaluate a model on MP-Bench from ASR transcripts (paper Table 3 metrics).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("detect", help="Turn-taking + latency + post-input text from timestamps")
    _add_common(p)
    p.add_argument("--spoke-tsv", default=None, help="Optional spoke_after_input_results.tsv")
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("judge", help="Run the paper LLM judges")
    _add_common(p)
    p.add_argument("--provider", choices=sorted(judge_mod.PROVIDERS), default="openai")
    p.add_argument("--model", default="gpt-5")
    p.add_argument("--api-key", default=None)
    p.add_argument("--kinds", nargs="+", choices=list(judge_mod.PROMPTS), default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--no-sampling-params", action="store_true")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_judge)

    p = sub.add_parser("report", help="Write Table 3 metrics")
    _add_common(p)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("run", help="detect + judge + report")
    _add_common(p)
    p.add_argument("--spoke-tsv", default=None)
    p.add_argument("--provider", choices=sorted(judge_mod.PROVIDERS), default="openai")
    p.add_argument("--model", default="gpt-5")
    p.add_argument("--api-key", default=None)
    p.add_argument("--kinds", nargs="+", choices=list(judge_mod.PROMPTS), default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--no-sampling-params", action="store_true")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_run)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
