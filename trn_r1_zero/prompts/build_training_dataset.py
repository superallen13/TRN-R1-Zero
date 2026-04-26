"""
Simplified training dataset builder using the same template as evaluation.

- Loads a cleaned TAG .pt via load_tag(path).
- Uses only the train mask to select nodes.
- Uses one-hop neighbor sampling (random, up to K neighbors).
- Produces a HuggingFace DatasetDict with only the 'train' split, keeping
  the same columns as evaluation: problem, solution, dataset_name, split, idx, system_prompt.

Optionally saves locally (save_to_disk) and/or pushes to the Hub.
"""
from __future__ import annotations

import os
import argparse
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, concatenate_datasets
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from trn_r1_zero.tag.core import load_tag
from trn_r1_zero.prompts.templates import SYSTEMS, INSTRUCTIONS
from trn_r1_zero.prompts.dataset_meta import RELATIONS, NODE_TYPES

PROMPT_COLUMNS = [
    "problem",
    "solution",
    "dataset_name",
    "split",
    "idx",
    "system_prompt",
    "hardness_score",
    "hardness_raw_gain",
    "hardness_abs_gain",
]


class SGCAugment(MessagePassing):
    """Shallow graph convolution used to diffuse sentence embeddings."""

    def __init__(self, num_layers: int = 2, add_self_loops: bool = True):
        super().__init__()
        self.num_layers = num_layers
        self.add_self_loops = add_self_loops

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight=None) -> torch.Tensor:
        edge_index, norm = gcn_norm(
            edge_index,
            edge_weight,
            x.size(0),
            improved=False,
            add_self_loops=self.add_self_loops,
        )
        for _ in range(self.num_layers):
            x = self.propagate(edge_index, x=x, norm=norm)
        return x

    def message(self, x_j: torch.Tensor, norm: torch.Tensor | None) -> torch.Tensor:  # type: ignore[override]
        return norm.view(-1, 1) * x_j if norm is not None else x_j


def mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def encode_texts(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    batch_size: int,
    device: torch.device,
    desc: str,
) -> torch.Tensor:
    embeddings: List[torch.Tensor] = []
    for start in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
        pooled = mean_pooling(outputs, encoded["attention_mask"])
        pooled = F.normalize(pooled, p=2, dim=1)
        embeddings.append(pooled.cpu().to(torch.float32))
    return torch.cat(embeddings, dim=0)


def compute_margin_gain_scores(
    tag,
    encoder_name: str,
    batch_size: int,
    sgc_layers: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    if getattr(tag, "raw_texts", None) is None:
        raise ValueError("TAG data does not contain 'raw_texts'; cannot compute hardness scores.")
    if getattr(tag, "label_name", None) is None:
        raise ValueError("TAG data does not contain 'label_name'; cannot compute hardness scores.")

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    model = AutoModel.from_pretrained(encoder_name).to(device)
    model.eval()

    label_texts = [str(name) for name in tag.label_name]
    node_texts = list(tag.raw_texts)

    label_embeddings = encode_texts(
        label_texts,
        tokenizer,
        model,
        batch_size=batch_size,
        device=device,
        desc="Encoding labels",
    ).to(device)
    node_embeddings = encode_texts(
        node_texts,
        tokenizer,
        model,
        batch_size=batch_size,
        device=device,
        desc="Encoding nodes",
    ).to(device)

    with torch.no_grad():
        raw_logits = node_embeddings @ label_embeddings.T

        if sgc_layers > 0:
            aggregator = SGCAugment(num_layers=sgc_layers)
            propagated = aggregator(node_embeddings, tag.edge_index.to(device))
            propagated = F.normalize(propagated, p=2, dim=1)
        else:
            propagated = node_embeddings

        aggregated_logits = propagated @ label_embeddings.T
        labels_device = tag.y.to(device)

        idx = torch.arange(labels_device.size(0), device=device)

        def multiclass_margin(logits: torch.Tensor) -> torch.Tensor:
            true_scores = logits[idx, labels_device]
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask[idx, labels_device] = True
            other_scores = logits.masked_fill(mask, float("-inf")).max(dim=1).values
            return true_scores - other_scores

        raw_margin = multiclass_margin(raw_logits)
        agg_margin = multiclass_margin(aggregated_logits)
        delta = agg_margin - raw_margin
        gain = torch.clamp(delta, min=0.0)
        abs_delta = delta.abs()

    return {
        "clamped": gain.cpu(),
        "raw": delta.cpu(),
        "abs": abs_delta.cpu(),
    }


def to_adj_list(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must be shape [2, E]")
    adj = [[] for _ in range(num_nodes)]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for u, v in zip(src, dst):
        if v not in adj[u]:
            adj[u].append(v)
        if u not in adj[v]:
            adj[v].append(u)
    return adj


def empty_prompt_dataset() -> Dataset:
    return Dataset.from_dict({col: [] for col in PROMPT_COLUMNS})


def format_labels(label_names: List[str]) -> str:
    return "\n".join([f"{i}: {n}" for i, n in enumerate(label_names)])


def truncate_text(text: str, max_words: Optional[int]) -> str:
    if max_words is None or max_words <= 0:
        return text.strip()
    words = text.strip().split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + " ..."


def render_context(target_text: str, neighbor_texts: List[str]) -> str:
    lines = ["Target node:", target_text.strip(), ""]
    if not neighbor_texts:
        lines.append("No neighbor nodes available.")
        return "\n".join(lines)
    lines.append("Neighbor nodes:")
    for i, t in enumerate(neighbor_texts, 1):
        lines.append(f"[{i}] {t.strip()}")
        lines.append("")
    return "\n".join(lines)


def pick_neighbors(adj: List[List[int]], node_id: int, k: int, rng: np.random.Generator, vary_k: bool = False) -> List[int]:
    """Randomly sample up to k 1-hop neighbors without replacement.
    - When degree <= k, return a random permutation of all neighbors.
    - If vary_k=True, randomly vary the subset size in [1, min(k, deg)] to add augmentation diversity.
    """
    neigh = adj[node_id]
    deg = len(neigh)
    if deg == 0:
        return []
    smax = min(k, deg)
    size = int(rng.integers(1, smax + 1)) if vary_k else smax
    idx = rng.choice(deg, size=size, replace=False)
    return [neigh[i] for i in idx]


def mask_to_indices(mask_tensor: torch.Tensor | None) -> List[int]:
    if mask_tensor is None:
        return []
    return torch.nonzero(mask_tensor, as_tuple=False).view(-1).tolist()


def build_one_problem(
    dataset_name: str,
    node_id: int,
    raw_texts: List[str],
    neighbors: List[int],
    relation: str,
    node_type_desc: str,
    labels_block: str,
    num_categories: int,
    max_text_words: Optional[int] = None,
) -> str:
    target_text = truncate_text(str(raw_texts[node_id]), max_text_words)
    neighbor_texts = [truncate_text(str(raw_texts[n]), max_text_words) for n in neighbors]
    context = render_context(target_text, neighbor_texts)
    instr_template = INSTRUCTIONS["v1"]
    question = instr_template.format(
        relation=relation,
        num_categories=num_categories,
        labels=labels_block,
        node_type=node_type_desc,
        max_id=num_categories - 1,
    )
    return f"{context}\n\n{question}"


def build_training_dataset(
    dataset_name: str,
    tag_path: str,
    k_neighbors: int = 3,
    seed: int = 1024,
    system_key: str = "simple",
    augmentations: int = 1,
    vary_k: bool = True,
    score_encoder: str = "sentence-transformers/all-MiniLM-L6-v2",
    score_batch_size: int = 32,
    score_sgc_layers: int = 1,
    score_device: Optional[str] = None,
    max_text_words: Optional[int] = None,
) -> DatasetDict:
    data = load_tag(tag_path)
    num_nodes = data.y.shape[0]
    num_classes = int(torch.max(data.y).item()) + 1

    label_names_raw = list(data.label_name)
    labels_block = format_labels(label_names_raw)

    adj = to_adj_list(data.edge_index, num_nodes)
    rng = np.random.default_rng(seed)

    relation = RELATIONS.get(dataset_name, "relation")
    node_type_desc = NODE_TYPES.get(dataset_name, "texts")
    system_prompt = SYSTEMS[system_key]

    # Only use train mask
    train_ids = torch.nonzero(data.train_mask, as_tuple=False).view(-1).tolist() if getattr(data, "train_mask", None) is not None else list(range(num_nodes))

    device = torch.device(score_device or ("cuda" if torch.cuda.is_available() else "cpu"))
    hardness_payload = compute_margin_gain_scores(
        tag=data,
        encoder_name=score_encoder,
        batch_size=score_batch_size,
        sgc_layers=score_sgc_layers,
        device=device,
    )
    hardness_scores_list = hardness_payload["clamped"].tolist()
    raw_gain_list = hardness_payload["raw"].tolist()
    abs_gain_list = hardness_payload["abs"].tolist()

    raw_texts = [str(x) for x in data.raw_texts]

    rows = {
        "problem": [],
        "solution": [],
        "dataset_name": [],
        "split": [],
        "idx": [],
        "system_prompt": [],
        "hardness_score": [],
        "hardness_raw_gain": [],
        "hardness_abs_gain": [],
    }
    for nid in train_ids:
        repeats = max(1, int(augmentations))
        for _ in range(repeats):
            neigh = pick_neighbors(adj, nid, k_neighbors, rng, vary_k=vary_k)
            problem = build_one_problem(
                dataset_name=dataset_name,
                node_id=nid,
                raw_texts=raw_texts,
                neighbors=neigh,
                relation=relation,
                node_type_desc=node_type_desc,
                labels_block=labels_block,
                num_categories=num_classes,
                max_text_words=max_text_words,
            )
            gt_label = int(data.y[nid].item())
            rows["problem"].append(problem)
            rows["solution"].append(str(gt_label))
            rows["dataset_name"].append(dataset_name)
            rows["split"].append("train")
            rows["idx"].append(int(nid))
            rows["system_prompt"].append(system_prompt)
            rows["hardness_score"].append(float(hardness_scores_list[nid]))
            rows["hardness_raw_gain"].append(float(raw_gain_list[nid]))
            rows["hardness_abs_gain"].append(float(abs_gain_list[nid]))

    ds_train = Dataset.from_dict(rows)
    return DatasetDict({"train": ds_train})


def build_eval_dataset(
    dataset_name: str,
    tag_path: str,
    k_neighbors: int = 3,
    seed: int = 1024,
    system_key: str = "simple",
    score_encoder: str = "sentence-transformers/all-MiniLM-L6-v2",
    score_batch_size: int = 32,
    score_sgc_layers: int = 1,
    score_device: Optional[str] = None,
    max_text_words: Optional[int] = None,
) -> DatasetDict:
    tag = load_tag(tag_path)
    num_nodes = tag.y.shape[0]
    num_classes = int(torch.max(tag.y).item()) + 1

    label_names_raw = list(tag.label_name)
    labels_block = format_labels(label_names_raw)

    adj = to_adj_list(tag.edge_index, num_nodes)
    rng = np.random.default_rng(seed)

    relation = RELATIONS.get(dataset_name, "relation")
    node_type_desc = NODE_TYPES.get(dataset_name, "texts")
    system_prompt = SYSTEMS[system_key]
    raw_texts = [str(x) for x in tag.raw_texts]

    device = torch.device(score_device or ("cuda" if torch.cuda.is_available() else "cpu"))
    hardness_payload = compute_margin_gain_scores(
        tag=tag,
        encoder_name=score_encoder,
        batch_size=score_batch_size,
        sgc_layers=score_sgc_layers,
        device=device,
    )
    hardness_scores_list = hardness_payload["clamped"].tolist()
    raw_gain_list = hardness_payload["raw"].tolist()
    abs_gain_list = hardness_payload["abs"].tolist()

    def deterministic_neighbors(node_id: int) -> List[int]:
        neighbors = adj[node_id]
        if len(neighbors) <= k_neighbors:
            return list(neighbors)
        idx = rng.choice(len(neighbors), size=k_neighbors, replace=False)
        return [neighbors[i] for i in idx]

    def make_rows(indices: List[int], split_name: str) -> Dict[str, List]:
        rows = {
            "problem": [],
            "solution": [],
            "dataset_name": [],
            "split": [],
            "idx": [],
            "system_prompt": [],
            "hardness_score": [],
            "hardness_raw_gain": [],
            "hardness_abs_gain": [],
        }
        iterator = tqdm(indices, desc=f"Building {dataset_name}:{split_name}", unit="nodes") if indices else []
        for nid in iterator:
            neigh = deterministic_neighbors(nid)
            problem = build_one_problem(
                dataset_name=dataset_name,
                node_id=nid,
                raw_texts=raw_texts,
                neighbors=neigh,
                relation=relation,
                node_type_desc=node_type_desc,
                labels_block=labels_block,
                num_categories=num_classes,
                max_text_words=max_text_words,
            )
            gt_label = int(tag.y[nid].item())
            rows["problem"].append(problem)
            rows["solution"].append(str(gt_label))
            rows["dataset_name"].append(dataset_name)
            rows["split"].append(split_name)
            rows["idx"].append(int(nid))
            rows["system_prompt"].append(system_prompt)
            rows["hardness_score"].append(float(hardness_scores_list[nid]))
            rows["hardness_raw_gain"].append(float(raw_gain_list[nid]))
            rows["hardness_abs_gain"].append(float(abs_gain_list[nid]))
        return rows

    train_ids = mask_to_indices(getattr(tag, "train_mask", None))
    val_ids = mask_to_indices(getattr(tag, "val_mask", None))
    test_ids = mask_to_indices(getattr(tag, "test_mask", None))

    ds_dict = {}
    if train_ids:
        ds_dict["train"] = Dataset.from_dict(make_rows(train_ids, "train"))
    if val_ids:
        ds_dict["validation"] = Dataset.from_dict(make_rows(val_ids, "validation"))
    if test_ids:
        ds_dict["test"] = Dataset.from_dict(make_rows(test_ids, "test"))

    # Ensure empty splits exist for consistency
    for split in ("train", "validation", "test"):
        if split not in ds_dict:
            ds_dict[split] = empty_prompt_dataset()

    return DatasetDict(ds_dict)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build prompt datasets (train/eval) from cleaned TAG .pt files")
    p.add_argument("--mode", choices=["train", "eval"], default="train", help="Build training-only dataset or full eval splits")
    p.add_argument("--datasets", nargs="+", required=True, help="Dataset aliases to process (e.g., cora history)")
    p.add_argument("--tag-root", default="./datasets", help="Directory holding <dataset_name>.pt")
    p.add_argument("--neighbors", type=int, default=3, help="Max number of 1-hop neighbors to include")
    p.add_argument("--augmentations", type=int, default=1, help="Number of variants per train node (data augmentation)")
    p.add_argument("--fix_k", action="store_true", help="Use exactly k neighbors when possible (no random k in [1..k])")
    p.add_argument("--seed", type=int, default=1024, help="RNG seed")
    p.add_argument("--system", default="simple", choices=list(SYSTEMS.keys()), help="System prompt key")
    p.add_argument("--score-encoder", default="sentence-transformers/all-MiniLM-L6-v2", help="Encoder used to compute hardness scores")
    p.add_argument("--score-batch-size", type=int, default=32, help="Batch size when encoding texts for hardness")
    p.add_argument("--score-sgc-layers", type=int, default=1, help="Number of SGC propagation layers for hardness scoring (0 disables)")
    p.add_argument("--score-device", default=None, help="Torch device for hardness scoring, e.g. 'cuda' or 'cpu'")
    p.add_argument("--max-text-words", type=int, default=None, help="Optional cap on number of words per node text")
    p.add_argument("--save-dir", default=None, help="If set, save locally to this directory")
    p.add_argument("--out-name", default=None, help="Output folder name when combining multiple datasets (default: 'multi')")
    return p.parse_args(args=argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    mode = args.mode
    names = args.datasets

    if mode == "train":
        train_parts = []
        for name in names:
            tag_path = os.path.join(args.tag_root, f"{name}.pt")
            ds_i = build_training_dataset(
                dataset_name=name,
                tag_path=tag_path,
                k_neighbors=args.neighbors,
                seed=args.seed,
                system_key=args.system,
                augmentations=args.augmentations,
                vary_k=(not args.fix_k),
                score_encoder=args.score_encoder,
                score_batch_size=args.score_batch_size,
                score_sgc_layers=args.score_sgc_layers,
                score_device=args.score_device,
                max_text_words=args.max_text_words,
            )
            train_parts.append(ds_i["train"])

        combined_train = train_parts[0] if len(train_parts) == 1 else concatenate_datasets(train_parts)
        ds_dict = DatasetDict({"train": combined_train})

        print(ds_dict)
        if len(ds_dict["train"]) > 0:
            ex = ds_dict["train"][0]
            print("\n=== Example from train ===")
            print("idx:", ex["idx"])
            print("solution:", ex["solution"])
            print("dataset_name:", ex["dataset_name"])
            print("problem (first 10000 chars):\n", ex["problem"][:10000], "...")

        out_name = args.out_name or (names[0] if len(names) == 1 else "multi")
        save_suffix = f"train_nei{args.neighbors}_prompts"
        if args.max_text_words:
            save_suffix += f"_maxw{args.max_text_words}"
    else:  # eval mode
        split_parts = {"train": [], "validation": [], "test": []}
        for name in names:
            tag_path = os.path.join(args.tag_root, f"{name}.pt")
            ds_i = build_eval_dataset(
                dataset_name=name,
                tag_path=tag_path,
                k_neighbors=args.neighbors,
                seed=args.seed,
                system_key=args.system,
                score_encoder=args.score_encoder,
                score_batch_size=args.score_batch_size,
                score_sgc_layers=args.score_sgc_layers,
                score_device=args.score_device,
                max_text_words=args.max_text_words,
            )
            for split in ("train", "validation", "test"):
                if len(ds_i[split]) > 0:
                    split_parts[split].append(ds_i[split])

        ds_dict = DatasetDict()
        for split, parts in split_parts.items():
            if len(parts) == 1:
                ds_dict[split] = parts[0]
            elif len(parts) > 1:
                ds_dict[split] = concatenate_datasets(parts)
            else:
                ds_dict[split] = empty_prompt_dataset()

        print(ds_dict)
        for split in ("train", "validation", "test"):
            if len(ds_dict[split]) > 0:
                ex = ds_dict[split][0]
                print(f"\n=== Example from {split} ===")
                print("idx:", ex["idx"])
                print("solution:", ex["solution"])
                print("dataset_name:", ex["dataset_name"])
                print("problem (first 10000 chars):\n", ex["problem"][:10000], "...")
                break

        out_name = args.out_name or (names[0] if len(names) == 1 else "multi")
        save_suffix = f"eval_nei{args.neighbors}_prompts"
        if args.max_text_words:
            save_suffix += f"_maxw{args.max_text_words}"

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        out_path = os.path.join(args.save_dir, f"{out_name}_{save_suffix}")
        ds_dict.save_to_disk(out_path)
        print(f"\n[SAVED] Dataset saved locally to: {out_path}")


if __name__ == "__main__":
    """
    Example usage:
    python -m trn_r1_zero.prompts.build_training_dataset --datasets cora --neighbors 3 --fix_k --augmentations 10 --save-dir ./datasets/prompts
    """
    main()
