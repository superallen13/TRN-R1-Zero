import argparse
import os
from typing import Optional, List

from datasets import load_dataset, load_from_disk, Dataset, DatasetDict, concatenate_datasets
from trn_r1_zero.prompts.templates import SYSTEMS


def load_hf_or_local(name_or_path: str) -> DatasetDict:
    """Load a DatasetDict from HF Hub or a local save_to_disk path."""
    if os.path.exists(name_or_path):
        dd = load_from_disk(name_or_path)
        if not isinstance(dd, DatasetDict):
            # Some save formats return a Dataset; wrap into a dict with 'train'
            return DatasetDict({"train": dd})
        return dd
    return load_dataset(name_or_path)


def to_verl_format(ds: Dataset, data_source: str, system_prompt_fallback: str, split_name: str) -> Dataset:
    """Map a prompt dataset to VERL-friendly records as parquet."""

    def process_fn(example, idx):
        question = example.get("problem") or example.get("content") or ""
        solution = example.get("solution") or example.get("label") or ""
        sys_prompt = example.get("system_prompt") or system_prompt_fallback
        hardness_score = example.get("hardness_score")
        hardness_raw = example.get("hardness_raw_gain")
        hardness_abs = example.get("hardness_abs_gain")
        extra_info = {
            "split": split_name,
            "index": idx,
        }
        if hardness_score is not None:
            try:
                extra_info["hardness_score"] = float(hardness_score)
            except (TypeError, ValueError):
                pass
        if hardness_raw is not None:
            try:
                extra_info["hardness_raw_gain"] = float(hardness_raw)
            except (TypeError, ValueError):
                pass
        if hardness_abs is not None:
            try:
                extra_info["hardness_abs_gain"] = float(hardness_abs)
            except (TypeError, ValueError):
                pass
        return {
            "data_source": data_source,
            "prompt": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": question},
            ],
            "ability": "nc",
            "reward_model": {"style": "rule", "ground_truth": str(solution)},
            "extra_info": extra_info,
        }

    return ds.map(function=process_fn, with_indices=True, remove_columns=ds.column_names)
    

def summarize_hardness(ds: Dataset, split_name: str) -> None:
    total = len(ds)
    if total == 0:
        print(f"[{split_name}] total=0 (empty split)")
        return
    counts = {
        "hardness_score": 0,
        "hardness_raw_gain": 0,
        "hardness_abs_gain": 0,
    }
    for item in ds:
        extra = item.get("extra_info", {})
        if isinstance(extra, dict):
            for key in list(counts.keys()):
                if extra.get(key) is not None:
                    counts[key] += 1
    for key, count in counts.items():
        pct = count / total * 100.0
        print(f"[{split_name}] {key} present: {count}/{total} ({pct:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Preprocess prompt datasets (HF or local) into VERL parquet format")
    # Mode 1: unified merge from explicit train/test lists (preferred for multi-domain)
    parser.add_argument("--train_datasets", nargs="+", default=None, help="List of dataset repos/paths to merge 'train' split from")
    parser.add_argument("--test_datasets", nargs="+", default=None, help="List of dataset repos/paths to merge 'test' split from")
    # Mode 2: legacy single dataset or multi-datasets single-split
    g = parser.add_mutually_exclusive_group(required=False)
    g.add_argument("--dataset_name", help="Single HF repo id or local save_to_disk path")
    g.add_argument("--datasets", nargs="+", help="List of dataset repos/paths to merge one split from")
    parser.add_argument("--out_dir", default="./verl_data", help="Output directory for parquet files")
    parser.add_argument("--splits", default="train,test", help="Comma-separated splits to export (single-dataset mode)")
    parser.add_argument("--split", default="test", help="Split to merge across datasets (multi-dataset mode)")
    parser.add_argument("--out_name", default=None, help="Output parquet base name for merged mode (default: split)")
    parser.add_argument("--system_prompt_key", default="simple", choices=list(SYSTEMS.keys()), help="Fallback system prompt key if not provided in data")
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    sys_fallback = SYSTEMS[args.system_prompt_key]

    # Preferred: explicit train/test lists for unified export
    if (args.train_datasets is not None) or (args.test_datasets is not None):
        # Merge train
        if args.train_datasets is not None:
            train_parts = []
            total_train = 0
            for name in args.train_datasets:
                dd = load_hf_or_local(name)
                if "train" not in dd:
                    print(f"Skip {name}: split 'train' not found. Available: {list(dd.keys())}")
                    continue
                raw_n = len(dd["train"])  # before mapping
                print(f"Select train: {name} -> {raw_n} samples")
                ds_out = to_verl_format(dd["train"], name, sys_fallback, "train")
                summarize_hardness(ds_out, f"{name}/train")
                train_parts.append(ds_out)
                total_train += len(ds_out)
            if train_parts:
                merged_train = concatenate_datasets(train_parts) if len(train_parts) > 1 else train_parts[0]
                summarize_hardness(merged_train, "merged_train")
                merged_train.to_parquet(os.path.join(out_dir, "train.parquet"))
                print(f"Merged train: {total_train} records -> {os.path.join(out_dir, 'train.parquet')}")
            else:
                print("No train datasets exported (none had a 'train' split).")

        # Merge test
        if args.test_datasets is not None:
            test_parts = []
            total_test = 0
            for name in args.test_datasets:
                dd = load_hf_or_local(name)
                if "test" not in dd:
                    print(f"Skip {name}: split 'test' not found. Available: {list(dd.keys())}")
                    continue
                raw_n = len(dd["test"])  # before mapping
                print(f"Select test:  {name} -> {raw_n} samples")
                ds_out = to_verl_format(dd["test"], name, sys_fallback, "test")
                summarize_hardness(ds_out, f"{name}/test")
                test_parts.append(ds_out)
                total_test += len(ds_out)
            if test_parts:
                merged_test = concatenate_datasets(test_parts) if len(test_parts) > 1 else test_parts[0]
                summarize_hardness(merged_test, "merged_test")
                merged_test.to_parquet(os.path.join(out_dir, "test.parquet"))
                print(f"Merged test: {total_test} records -> {os.path.join(out_dir, 'test.parquet')}")
            else:
                print("No test datasets exported (none had a 'test' split).")

        # Done in explicit mode
        return

    # Multi-dataset merge mode (single split)
    if args.datasets is not None:
        split = args.split
        parts = []
        total = 0
        for name in args.datasets:
            dd = load_hf_or_local(name)
            if split not in dd:
                print(f"Skip {name}: split '{split}' not found. Available: {list(dd.keys())}")
                continue
            ds_out = to_verl_format(dd[split], name, sys_fallback, split)
            parts.append(ds_out)
            total += len(ds_out)
        if not parts:
            raise SystemExit("No datasets had the requested split to merge.")
        merged = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
        base = args.out_name or split
        out_path = os.path.join(out_dir, f"{base}.parquet")
        merged.to_parquet(out_path)
        print(f"Merged {len(args.datasets)} datasets, split='{split}': {total} records -> {out_path}")
        return

    # Single-dataset export mode
    data_source = args.dataset_name
    dd = load_hf_or_local(data_source)
    wanted_splits: List[str] = [s.strip() for s in args.splits.split(",") if s.strip()]
    exported = []
    for split in wanted_splits:
        if split not in dd:
            continue
        ds_out = to_verl_format(dd[split], data_source, sys_fallback, split)
        out_path = os.path.join(out_dir, f"{split}.parquet")
        ds_out.to_parquet(out_path)
        exported.append((split, len(ds_out)))
    if not exported:
        raise SystemExit(f"No requested splits found in dataset. Available: {list(dd.keys())}")
    for split, n in exported:
        print(f"Exported {split}: {n} records -> {os.path.join(out_dir, f'{split}.parquet')}")


if __name__ == "__main__":
    """
    Example usage:
    python -m trn_r1_zero.prompts.data_preprocessing \
        --train_datasets ./datasets/prompts/citeseer_train_nei3_prompts ./datasets/prompts/history_train_nei3_prompts \
        --test_datasets  ./datasets/prompts/citeseer_eval_nei3_prompts  ./datasets/prompts/cora_eval_nei3_prompts \
        --out_dir ./verl_data/run
    """
    main()
