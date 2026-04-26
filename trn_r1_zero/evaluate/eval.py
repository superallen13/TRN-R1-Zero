"""
Async vLLM client for evaluating TRN-R1-Zero models.

Spawns N async workers against one or more local vLLM OpenAI-compatible
servers. Each worker pulls examples from a shared queue, posts a chat
completion, extracts the answer from `<answer>...</answer>`, and accumulates
per-group accuracy + macro F1 + hallucination rate.

The vLLM server(s) should already be running (see scripts/eval/eval.sh).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import requests
from datasets import DatasetDict, load_dataset, load_from_disk
from openai import AsyncOpenAI
from tqdm import tqdm

from trn_r1_zero.prompts.templates import SYSTEMS

# ---- Defaults ---------------------------------------------------------------
MODEL_NAME = "Allen-UQ/trn-r1-zero-7b"
DATASET_NAME = "Allen-UQ/trn-r1-zero-eval-ds-cora"
DATASET_SPLIT = "all"
DEFAULT_BASE_PORT = 21000
TEMPERATURE = 0.0
TOP_P = 1.0
SEED = 1024
MAX_STARTUP_WAIT = 1200
QUERY_TIMEOUT = 300
JSON_WRITE_INTERVAL = 10
ACC_UPDATE_INTERVAL = 10

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


# ---- vLLM helpers -----------------------------------------------------------

def wait_for_server(port: int, timeout: int = MAX_STARTUP_WAIT) -> None:
    url = f"http://localhost:{port}/v1/models"
    print(f"Waiting for vLLM server on port {port}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            if requests.get(url, timeout=5).status_code == 200:
                print(f"vLLM server on port {port} is ready ✅")
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(10)
    raise TimeoutError(f"vLLM server on port {port} did not start within {timeout}s")


# ---- Output formatting helpers ---------------------------------------------

def extract_answer(completion: str) -> str:
    """Pull `<answer>...</answer>`; fall back to the full completion."""
    m = ANSWER_RE.search(completion)
    return m.group(1).strip() if m else completion.strip()


def make_model_tag(name_or_path: str) -> str:
    base = Path(name_or_path).name if os.path.exists(name_or_path) else name_or_path.split("/")[-1]
    s = re.sub(r"(\d)\.(\d)", r"\1_\2", base)
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).lower()
    return re.sub(r"_+", "_", s).strip("_")


def output_base(name_or_path: str) -> str:
    if os.path.exists(name_or_path):
        return Path(name_or_path).name
    parts = [p for p in name_or_path.split("/") if p]
    return parts[-1] if parts else name_or_path


# ---- Dataset loading --------------------------------------------------------

def load_hf_dataset(name_or_path: str, split: str):
    """Load a DatasetDict from the Hub or a local save_to_disk path. Supports split='all'."""
    split_lower = split.lower()
    if os.path.exists(name_or_path):
        dd = load_from_disk(name_or_path)
        if split_lower == "all":
            if not isinstance(dd, DatasetDict):
                raise ValueError(f"{name_or_path} is not a DatasetDict; cannot use split=all")
            return dd
        norm = {"val": "validation", "valid": "validation"}.get(split_lower, split)
        if isinstance(dd, DatasetDict):
            if norm not in dd:
                raise ValueError(f"Split '{split}' not in {name_or_path}; available: {list(dd.keys())}")
            return dd[norm]
        raise ValueError(f"{name_or_path} is not a DatasetDict")

    if split_lower == "all":
        ds = load_dataset(name_or_path)
        return ds if isinstance(ds, DatasetDict) else DatasetDict({"all": ds})
    return load_dataset(name_or_path, split=split)


def normalize_examples(ds_hf, default_split: str) -> List[Dict[str, Any]]:
    rows = []
    for ex in ds_hf:
        node_id = ex.get("node_id")
        if node_id is None:
            node_id = ex.get("idx")
        rows.append(
            {
                "content": ex.get("content") or ex.get("problem") or ex.get("prompt") or "",
                "label": str(ex.get("label") or ex.get("solution") or ""),
                "dataset": ex.get("dataset") or ex.get("dataset_name") or "default",
                "node_id": node_id,
                "split": ex.get("split") or default_split,
            }
        )
    return rows


# ---- Async query ------------------------------------------------------------

async def query_model(
    client: AsyncOpenAI, content: str, model_name: str, system_prompt: str
) -> Dict[str, Any]:
    messages: List[Dict[str, Any]] = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})

    try:
        completion = await asyncio.wait_for(
            client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                seed=SEED,
                stream=False,
            ),
            timeout=QUERY_TIMEOUT,
        )
        text = completion.choices[0].message.content
        try:
            raw = completion.model_dump()
        except AttributeError:
            raw = {}
        return {"content": text, "raw": raw}
    except asyncio.TimeoutError:
        return {"content": "TIMEOUT", "raw": {"error": "timeout"}}
    except Exception as e:
        return {"content": f"ERROR: {e}", "raw": {"error": str(e)}}


# ---- Evaluation loop --------------------------------------------------------

async def evaluate(
    clients: List[AsyncOpenAI],
    dataset: List[Dict[str, Any]],
    model_name: str,
    system_prompt: str,
    post_process_fn: Callable[[str], str],
    output_file: str,
    num_workers: int,
    capture_completion_count: int,
    report_mode: str,
):
    results: Dict[str, Dict[Any, Any]] = defaultdict(dict)
    correct: Dict[str, int] = defaultdict(int)
    total: Dict[str, int] = defaultdict(int)
    hallucination: Dict[str, int] = defaultdict(int)
    preds_by_group: Dict[str, List[str]] = defaultdict(list)
    labels_by_group: Dict[str, List[str]] = defaultdict(list)
    valid_labels: Dict[str, set] = defaultdict(set)

    def group_key(item: Dict[str, Any]) -> str:
        if report_mode == "split":
            return str(item.get("split", "unknown"))
        if report_mode == "all":
            return "all"
        return str(item.get("dataset", "default"))

    for item in dataset:
        valid_labels[group_key(item)].add(str(item.get("label", "")).strip().lower())

    queue: asyncio.Queue = asyncio.Queue()
    for idx, item in enumerate(dataset):
        await queue.put((idx, item))

    pbar = tqdm(total=len(dataset), desc="Evaluating", unit=" samples")
    processed = 0

    def keep_completion(node_id: Any, idx: int) -> bool:
        if capture_completion_count <= 0:
            return False
        try:
            return int(node_id) < capture_completion_count
        except (TypeError, ValueError):
            return idx < capture_completion_count

    async def worker(client: AsyncOpenAI):
        nonlocal processed
        while True:
            try:
                idx, item = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if queue.empty():
                    break
                continue

            payload = await query_model(client, item["content"], model_name, system_prompt)
            raw_text = payload["content"]
            pred = post_process_fn(raw_text)

            ds_name = item["dataset"]
            node_id = item["node_id"]
            label = str(item["label"]).strip()
            split = item["split"]

            results[ds_name][node_id] = {
                "node_id": node_id,
                "processed_result": pred,
                "label": label,
                "split": split,
                "completion": raw_text if keep_completion(node_id, idx) else None,
                "raw_completion": payload["raw"] if keep_completion(node_id, idx) else None,
            }

            gk = group_key(item)
            total[gk] += 1
            if pred.lower() == label.lower():
                correct[gk] += 1
            preds_by_group[gk].append(pred.lower())
            labels_by_group[gk].append(label.lower())
            if pred.lower() not in valid_labels[gk]:
                hallucination[gk] += 1

            processed += 1
            if processed % JSON_WRITE_INTERVAL == 0:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)

            queue.task_done()
            pbar.update(1)
            if processed % ACC_UPDATE_INTERVAL == 0 or processed == len(dataset):
                acc_str = {
                    ds: f"{(correct[ds] / total[ds] * 100):.2f}%" if total[ds] else "0.00%"
                    for ds in total
                }
                pbar.set_postfix(acc=str(acc_str))

    tasks = [asyncio.create_task(worker(clients[i % len(clients)])) for i in range(num_workers)]
    await queue.join()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    pbar.close()

    macro_f1: Dict[str, float] = {}
    for group, preds in preds_by_group.items():
        golds = labels_by_group[group]
        if not (preds and golds):
            macro_f1[group] = 0.0
            continue
        classes = sorted(set(golds) | set(preds))
        f1s = []
        for c in classes:
            tp = sum(1 for p, g in zip(preds, golds) if p == c and g == c)
            fp = sum(1 for p, g in zip(preds, golds) if p == c and g != c)
            fn = sum(1 for p, g in zip(preds, golds) if g == c and p != c)
            if tp + fp == 0 or tp + fn == 0:
                f1s.append(0.0)
                continue
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
        macro_f1[group] = float(np.mean(f1s)) if f1s else 0.0

    return correct, total, hallucination, valid_labels, macro_f1


# ---- CLI --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Async vLLM evaluation for TRN-R1-Zero")
    parser.add_argument("--model_name", default=MODEL_NAME, help="HF model id or local path")
    parser.add_argument("--dataset_name", default=DATASET_NAME, help="HF DatasetDict id or local path")
    parser.add_argument("--dataset_split", default=DATASET_SPLIT, help="Split name or 'all'")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of vLLM servers (one per GPU)")
    parser.add_argument("--base_port", type=int, default=DEFAULT_BASE_PORT)
    parser.add_argument("--concurrency_factor", type=int, default=8, help="Async workers per server")
    parser.add_argument(
        "--system_prompt_key", default="simple", choices=list(SYSTEMS.keys())
    )
    parser.add_argument("--model_tag", default=None, help="Override the auto-derived model tag")
    parser.add_argument("--report_mode", default="dataset", choices=["dataset", "split", "all"])
    parser.add_argument(
        "--capture_completion_count",
        type=int,
        default=100,
        help="Keep full completions for the first N node ids (0 disables)",
    )
    args = parser.parse_args()

    num_gpus = max(1, min(8, args.num_gpus))
    num_workers = num_gpus * max(1, args.concurrency_factor)
    print(f"vLLM servers: {num_gpus}  async workers: {num_workers}")

    for i in range(num_gpus):
        wait_for_server(args.base_port + i)
    clients = [
        AsyncOpenAI(base_url=f"http://localhost:{args.base_port + i}/v1", api_key="dummy")
        for i in range(num_gpus)
    ]

    system_prompt = SYSTEMS.get(args.system_prompt_key, SYSTEMS["simple"])

    ds = load_hf_dataset(args.dataset_name, args.dataset_split)
    if isinstance(ds, DatasetDict):
        rows = []
        for split_name, subset in ds.items():
            rows.extend(normalize_examples(subset, split_name))
    else:
        rows = normalize_examples(ds, args.dataset_split)

    out_base = output_base(args.dataset_name)
    model_tag = args.model_tag or make_model_tag(args.model_name)
    eval_dir = os.path.join(
        os.environ.get("RESULTS_DIR", "results"), "eval", out_base, args.dataset_split
    )
    os.makedirs(eval_dir, exist_ok=True)
    output_file = os.path.join(eval_dir, f"{out_base}_{args.dataset_split}_{model_tag}.result.json")

    correct, total, halluc, valid, macro_f1 = asyncio.run(
        evaluate(
            clients,
            rows,
            args.model_name,
            system_prompt,
            extract_answer,
            output_file,
            num_workers,
            max(0, args.capture_completion_count),
            args.report_mode,
        )
    )

    print(f"\n✅ Results saved to {output_file}")

    summary = [f"\n📊 Evaluation metrics (grouped by {args.report_mode}):"]
    for g in sorted(total):
        acc = correct[g] / total[g] * 100 if total[g] else 0.0
        f1 = macro_f1.get(g, 0.0)
        hall = halluc[g] / total[g] * 100 if total[g] else 0.0
        label = {"all": "All samples", "split": f"Split '{g}'"}.get(
            args.report_mode, f"Dataset '{g}'"
        )
        summary += [
            f"  - {label}:",
            f"    * Accuracy: {acc:.2f}% ({correct[g]}/{total[g]})",
            f"    * Macro F1: {f1 * 100:.2f}%",
            f"    * Hallucination Rate: {hall:.2f}% ({halluc[g]}/{total[g]})",
            f"    * Valid Labels: {sorted(valid[g])}",
        ]
    print("\n".join(summary))

    summary_file = output_file.replace(".result.json", ".result")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("\n".join(summary))
    print(f"\n✅ Summary saved to {summary_file}")


if __name__ == "__main__":
    main()
