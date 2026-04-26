"""
Central registry for system prompts and instruction templates.
"""

from typing import Dict

# --- System prompts ---
SYSTEM_SIMPLE = (
    "You are a helpful AI Assistant that provides well-reasoned and detailed responses. "
    "You first think about the reasoning process as an internal monologue and then provide the user with the answer. "
    "Respond in the following format:\n<think>\n...\n</think>\n<answer>\n...\n</answer>"
)

SYSTEMS: Dict[str, str] = {
    "simple": SYSTEM_SIMPLE,
    "none": "",
}

# --- Instruction templates ---
# Use Python's format() keys: relation, num_categories, labels, node_type, max_id
INSTRUCTION_V1 = (
    "I provide the content of the target node and its neighbor nodes. "
    "Each node content is {node_type}. "
    "The relation between the target node and its neighbor nodes is {relation}. "
    "The {num_categories} categories are:\n{labels}\n"
    "Question: Based on the information of the target and neighbor nodes, "
    "predict the category ID (0 to {max_id}) for the target node."
)

INSTRUCTIONS: Dict[str, str] = {
    "v1": INSTRUCTION_V1,
}
