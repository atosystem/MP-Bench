"""LLM-as-a-judge clients for the four paper prompts (GPT-5 by default)."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .benchmark import PACKAGE_ROOT, Task

PROMPTS = {
    "understanding": PACKAGE_ROOT / "prompts" / "understanding.txt",
    "discussion": PACKAGE_ROOT / "prompts" / "discussion.txt",
    "turnbased": PACKAGE_ROOT / "prompts" / "turnbased.txt",
    "speaker_name": PACKAGE_ROOT / "prompts" / "speaker_name.txt",
}
SCHEMAS = {name: path.with_suffix(".schema.json") for name, path in PROMPTS.items()}
VERDICT_KEY = {
    "understanding": "correct",
    "speaker_name": "correct",
    "discussion": "consistent",
    "turnbased": "consistent",
}

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
MAX_ATTEMPTS = 5


def resolve_api_key(provider: str, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    names = {"openai": ("OPENAI_API_KEY",), "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY")}[provider]
    for name in names:
        if os.environ.get(name):
            return os.environ[name]
    raise SystemExit(f"Set {' or '.join(names)}, or pass --api-key")


def _post(url: str, body: dict, headers: dict, timeout: int) -> dict:
    request = Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _call_openai(model, api_key, prompt, schema, cfg) -> str:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": cfg["max_tokens"],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "mpbench_eval", "strict": True, "schema": schema},
        },
    }
    if cfg["temperature"] is not None:
        body["temperature"] = cfg["temperature"]
        body["top_p"] = cfg["top_p"]
    result = _post(OPENAI_URL, body, {"Authorization": f"Bearer {api_key}"}, cfg["timeout"])
    return result["choices"][0]["message"]["content"]


def _gemini_schema(node):
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key == "additionalProperties":
                continue
            if key == "type" and isinstance(value, str):
                out[key] = value.upper()
            elif key == "type" and isinstance(value, list):
                out[key] = value[0].upper() if value else "STRING"
            else:
                out[key] = _gemini_schema(value)
        return out
    if isinstance(node, list):
        return [_gemini_schema(item) for item in node]
    return node


def _call_gemini(model, api_key, prompt, schema, cfg) -> str:
    generation = {
        "maxOutputTokens": cfg["max_tokens"],
        "responseMimeType": "application/json",
        "responseSchema": _gemini_schema(schema),
    }
    if cfg["temperature"] is not None:
        generation["temperature"] = cfg["temperature"]
        generation["topP"] = cfg["top_p"]
    result = _post(
        GEMINI_URL.format(model=model),
        {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": generation},
        {"x-goog-api-key": api_key},
        cfg["timeout"],
    )
    return result["candidates"][0]["content"]["parts"][0]["text"]


PROVIDERS = {"openai": _call_openai, "gemini": _call_gemini}


def build_prompt(kind: str, task: Task, response: str) -> str:
    template = PROMPTS[kind].read_text(encoding="utf-8")
    if kind == "understanding":
        return template.format(
            metainformation=task.metatranscript,
            question_text=task.question_text,
            answer_text=task.answer_text,
            llm_predicted_answer=response,
        )
    if kind == "speaker_name":
        return template.format(
            question=task.question_text,
            gt_name=task.answer_text,
            response=response,
        )
    return template.format(metatranscript=task.metatranscript, llm_predicted_answer=response)


def _judge_one(call, model, api_key, prompt, schema, verdict_key, cfg):
    for attempt in range(MAX_ATTEMPTS):
        try:
            raw = call(model, api_key, prompt, schema, cfg)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            if exc.code in (408, 429, 500, 502, 503, 504) and attempt < MAX_ATTEMPTS - 1:
                time.sleep(5 * (attempt + 1))
                continue
            return None, f"HTTP {exc.code}: {detail}"
        except (URLError, TimeoutError, OSError) as exc:
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(2 * (attempt + 1))
                continue
            return None, f"{type(exc).__name__}: {exc}"
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None, f"unparseable: {raw[:300]}"
        if not isinstance(parsed.get(verdict_key), bool):
            return None, f"missing {verdict_key}: {raw[:300]}"
        return parsed, None
    return None, "retries exhausted"


def run_judge(
    kind: str,
    items: list[dict],
    tasks: dict[str, Task],
    provider: str,
    model: str,
    api_key: str,
    workers: int = 8,
    temperature: float | None = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 4096,
    timeout: int = 180,
    on_result=None,
) -> list[dict]:
    schema = json.loads(SCHEMAS[kind].read_text(encoding="utf-8"))
    call = PROVIDERS[provider]
    verdict_key = VERDICT_KEY[kind]
    cfg = {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens, "timeout": timeout}

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for item in items:
            prompt = build_prompt(kind, tasks[item["task_dir"]], item["response"])
            futures[
                pool.submit(_judge_one, call, model, api_key, prompt, schema, verdict_key, cfg)
            ] = item
        for future in as_completed(futures):
            item = futures[future]
            parsed, error = future.result()
            record = {
                "task_dir": item["task_dir"],
                "family": item.get("family"),
                "kind": kind,
                "response": item["response"],
                "verdict": parsed.get(verdict_key) if parsed else None,
                "reasoning": parsed.get("reasoning") if parsed else None,
                "error": error,
                "judge_provider": provider,
                "judge_model": model,
            }
            results.append(record)
            if on_result:
                on_result(record)
    results.sort(key=lambda r: r["task_dir"])
    return results
