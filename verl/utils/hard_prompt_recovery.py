"""Utilities for targeted recovery of all-zero rollout groups."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy


DEFAULT_PROCESS_HINT = (
    "Before giving the final answer, identify the relevant quantities, choose the required "
    "arithmetic operations, and calculate them step by step."
)


def all_zero_uids(uid_to_values: Mapping[object, Sequence[float]]) -> list[object]:
    return [uid for uid, values in uid_to_values.items() if values and all(float(value) == 0.0 for value in values)]


def source_indices_for_uids(source_uids: Sequence[object], selected_uids: Sequence[object]) -> list[int]:
    selected = set(selected_uids)
    return [index for index, uid in enumerate(source_uids) if uid in selected]


def add_process_hint(messages: object, hint: str = DEFAULT_PROCESS_HINT) -> list[dict]:
    if isinstance(messages, str):
        raise ValueError("raw_prompt must be a sequence of message mappings")
    try:
        copied = [deepcopy(dict(message)) for message in messages if isinstance(message, Mapping)]
    except TypeError as exc:
        raise ValueError("raw_prompt must be a sequence of message mappings") from exc
    if not copied:
        raise ValueError("raw_prompt has no messages")

    for message in copied:
        if message.get("role") == "system":
            message["content"] = f"{str(message.get('content', '')).rstrip()}\n{hint}"
            break
    else:
        copied.insert(0, {"role": "system", "content": hint})
    return copied
