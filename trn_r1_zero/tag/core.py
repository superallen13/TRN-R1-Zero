# trn_r1_zero/tag/core.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import torch
from torch import Tensor

try:
    # Optional: only needed if you want to run PyG baselines.
    from torch_geometric.data import Data as PyGData  # type: ignore
except Exception:
    PyGData = None


@dataclass
class TagData:
    """
    Minimal TAG container (pure PyTorch first, PyG optional via to_pyg()).

    Expected schema (the format released as Allen-UQ/trn-r1-zero-tags):
        - x: Tensor                   [N, D]
        - edge_index: Tensor          [2, E], long, undirected, deduplicated, no self-loops
        - y: Tensor                   [N], long
        - raw_texts: List[str]        (len N)
        - label_name: List[str]       (len C)
        - train_mask/val_mask/test_mask: Tensor [N], bool
    """
    x: Optional[Tensor]
    edge_index: Tensor
    y: Optional[Tensor] = None
    raw_texts: Optional[List[str]] = None
    label_name: Optional[List[str]] = None
    train_mask: Optional[Tensor] = None
    val_mask: Optional[Tensor] = None
    test_mask: Optional[Tensor] = None

    # ---- Basic properties ----
    def num_nodes(self) -> int:
        if self.x is not None:
            return self.x.size(0)
        if self.y is not None:
            return self.y.numel()
        # Infer from edge_index if no node features/labels
        return int(self.edge_index.max().item()) + 1 if self.edge_index.numel() > 0 else 0

    def num_classes(self) -> Optional[int]:
        if self.y is None:
            return None
        return int(self.y.max().item()) + 1 if self.y.numel() > 0 else None

    # ---- Device helpers ----
    def to(self, device: torch.device | str) -> "TagData":
        """Move tensor fields to device; texts remain on CPU."""
        dev = torch.device(device)
        return TagData(
            x=self.x.to(dev) if self.x is not None else None,
            edge_index=self.edge_index.to(dev),
            y=self.y.to(dev) if self.y is not None else None,
            raw_texts=self.raw_texts,
            label_name=self.label_name,
            train_mask=self.train_mask.to(dev) if self.train_mask is not None else None,
            val_mask=self.val_mask.to(dev) if self.val_mask is not None else None,
            test_mask=self.test_mask.to(dev) if self.test_mask is not None else None,
        )

    # ---- Split helpers ----
    def split_indices(self, split: str) -> Tensor:
        """Return node indices for one of {'train','val','test'}."""
        mask_map = {
            "train": self.train_mask,
            "val": self.val_mask,
            "test": self.test_mask,
        }
        mask = mask_map.get(split)
        if mask is None:
            raise ValueError(f"Split '{split}' not recognized or mask is not available.")
        return mask.nonzero(as_tuple=False).view(-1)

    # ---- Graph utilities ----
    def to_sparse_adj(self) -> torch.Tensor:
        """Return COO sparse adjacency with unit weights. Shape [N, N]."""
        N = self.num_nodes()
        idx = self.edge_index
        vals = torch.ones(idx.size(1), dtype=torch.float32, device=idx.device)
        return torch.sparse_coo_tensor(idx, vals, size=(N, N))

    def neighbors(self, node_idx: int) -> List[int]:
        """Return unique 1-hop neighbors (edge_index is already undirected)."""
        src, dst = self.edge_index[0], self.edge_index[1]
        nbrs = torch.cat([dst[src == node_idx], src[dst == node_idx]], dim=0).unique()
        return [int(i) for i in nbrs.tolist() if int(i) != node_idx]

    def neighbor_texts(self, node_idx: int) -> List[str]:
        """Return neighbor texts for a node; empty if texts not available."""
        if self.raw_texts is None:
            return []
        return [self.raw_texts[i] for i in self.neighbors(node_idx)]

    # ---- Optional adapter to PyG ----
    def to_pyg(self) -> "PyGData":
        """Convert to torch_geometric.data.Data if PyG is installed."""
        if PyGData is None:
            raise ImportError("torch_geometric is not installed.")
        data = PyGData()
        if self.x is not None:
            data.x = self.x
        data.edge_index = self.edge_index
        data.y = self.y
        data.train_mask = self.train_mask
        data.val_mask = self.val_mask
        data.test_mask = self.test_mask
        if self.raw_texts is not None:
            data.raw_texts = self.raw_texts
        if self.label_name is not None:
            data.label_name = self.label_name
        return data


def load_tag(path: str) -> TagData:
    """
    Load a cleaned TAG file (plain dict or PyG Data) and wrap it as TagData.
    The file MUST follow the schema released as Allen-UQ/trn-r1-zero-tags.
    """
    obj: Dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)

    def _get(k: str, default=None):
        if isinstance(obj, dict):
            return obj.get(k, default)
        return getattr(obj, k, default)

    x = _get("x", None)
    edge_index = _get("edge_index")
    y = _get("y")
    # Accept both our canonical schema and the TSGFM/sea_slime variant
    # (node_texts/label_names) so externally produced .pt files load too.
    raw_texts = _get("raw_texts", None) or _get("node_texts", None)
    label_name = _get("label_name", None) or _get("label_names", None)
    train_mask = _get("train_mask")
    val_mask = _get("val_mask")
    test_mask = _get("test_mask")

    # Sanity checks (fail fast, helpful error messages)
    assert isinstance(edge_index, torch.Tensor), "edge_index must be a Tensor"
    assert edge_index.dtype == torch.long and edge_index.dim() == 2 and edge_index.size(0) == 2, \
        "edge_index must be LongTensor of shape [2, E]"
    assert isinstance(y, torch.Tensor) and y.dim() == 1, "y must be LongTensor [N]"
    N = int(y.numel())
    for name, m in [("train_mask", train_mask), ("val_mask", val_mask), ("test_mask", test_mask)]:
        assert isinstance(m, torch.Tensor) and m.dtype == torch.bool and int(m.numel()) == N, \
            f"{name} must be BoolTensor [N]"
    if x is not None:
        assert x.size(0) == N, "x first dimension must equal N"
    if raw_texts is not None:
        assert len(raw_texts) == N, "len(raw_texts) must equal N"

    return TagData(
        x=x,
        edge_index=edge_index,
        y=y.long(),
        raw_texts=raw_texts,
        label_name=label_name,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


class NodeSplitDataset(torch.utils.data.Dataset):
    """
    Minimal node-level dataset over a given split ('train'/'val'/'test').
    Returns dicts so you can flexibly build prompts or feed tensors.
    """
    def __init__(self, tag: TagData, split: str = "train"):
        assert split in ("train", "val", "test")
        self.tag = tag
        self.indices = tag.split_indices(split)

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, i: int) -> Dict[str, Any]:
        idx = int(self.indices[i])
        item: Dict[str, Any] = {
            "idx": idx,
            "y": int(self.tag.y[idx].item()),
        }
        if self.tag.x is not None:
            item["x"] = self.tag.x[idx]          # [D]
        if self.tag.raw_texts is not None:
            item["text"] = self.tag.raw_texts[idx]
        return item


def default_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Simple collate for NodeSplitDataset:
        - stacks 'x' if present
        - keeps 'text' as list[str]
        - stacks 'y' as LongTensor
        - returns 'idx' as LongTensor
    """
    out: Dict[str, Any] = {}
    out["idx"] = torch.tensor([b["idx"] for b in batch], dtype=torch.long)
    out["y"] = torch.tensor([b["y"] for b in batch], dtype=torch.long)
    if "x" in batch[0]:
        out["x"] = torch.stack([b["x"] for b in batch], dim=0)  # [B, D]
    if "text" in batch[0]:
        out["text"] = [b["text"] for b in batch]                # List[str]
    return out

@dataclass
class TagGraphData(TagData):
    """
    Extends TagData for graph-level tasks.
    A collection of these objects represents a graph dataset.
    """
    graph_y: Optional[Tensor] = None              # [1] or scalar
    graph_raw_text: Optional[str] = None          # A single string for the graph
    edge_raw_texts: Optional[List[str]] = None    # List of strings for edges

    def to(self, device: torch.device | str) -> "TagGraphData":
        """Move tensor fields to device; texts remain on CPU."""
        base = super().to(device)
        # Create a new dictionary from the base object's __dict__
        base_dict = {k: v for k, v in base.__dict__.items() if k in TagData.__dataclass_fields__}
        return TagGraphData(
            **base_dict,
            graph_y=self.graph_y.to(device) if self.graph_y is not None else None,
            graph_raw_text=self.graph_raw_text,
            edge_raw_texts=self.edge_raw_texts,
        )

# ----------------------------
# Simple test (run: python trn_r1_zero/tag/core.py)
# ----------------------------

if __name__ == "__main__":
    import os
    from torch.utils.data import DataLoader
    # Example: change this to any dataset you have downloaded
    sample_path = os.path.join("datasets", "tags", "cora.pt")
    if not os.path.exists(sample_path):
        print(f"[WARN] {sample_path} not found. Download Allen-UQ/trn-r1-zero-tags first (see README).")
    else:
        tag = load_tag(sample_path)
        print("=== Loaded TAG dataset ===")
        print(f"Nodes: {tag.num_nodes()}, Edges: {tag.edge_index.size(1)}, Classes: {tag.num_classes()}")
        if tag.train_mask is not None:
            print(f"Train/Val/Test: {tag.train_mask.sum().item()} / "
                  f"{tag.val_mask.sum().item()} / {tag.test_mask.sum().item()}")
        # Show a few neighbors
        for i in range(3):
            nbs = tag.neighbors(i)
            print(f"Node {i}: label={int(tag.y[i])}, neighbors={nbs[:5]}...")
        # Test DataLoader with default_collate_fn
        train_set = NodeSplitDataset(tag, split="train")
        loader = DataLoader(train_set, batch_size=4, shuffle=True,
                            collate_fn=default_collate_fn)
        batch = next(iter(loader))
        print("=== One training batch ===")
        print({k: (v if isinstance(v, list) else v.shape) for k, v in batch.items()})




