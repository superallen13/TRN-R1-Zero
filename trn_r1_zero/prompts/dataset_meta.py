"""
Dataset-specific metadata (relations and node-type descriptors).
"""

from typing import Dict

# Relation between the target node and its 1-hop neighbor nodes
RELATIONS: Dict[str, str] = {
    "cora": "citation",
    "citeseer": "citation",
    "wikics": "hyperlink",
    "photo": "co-purchase",
    "history": "co-purchase",
    "instagram": "following",
}

# How to refer to the node texts in natural language inside the prompt
NODE_TYPES: Dict[str, str] = {
    "cora": "paper segment",
    "citeseer": "paper segment",
    "wikics": "wikipedia article",
    "photo": "customer review",
    "history": "product description",
    "instagram": "user bio",
}
