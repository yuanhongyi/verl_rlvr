"""Utilities for targeted recovery of all-zero rollout groups."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy


DEFAULT_PROCESS_HINT = (
    "Re-solve the problem using a different approach, but keep the calculation brief. "
    "Reserve enough tokens for the answer. Even if the reasoning is incomplete, end with "
    "your best numeric answer on a final line exactly `#### <number>`, and write nothing "
    "after it."
)


def validate_recovery_response_length(value: object, max_response_length: int) -> int | None:
    """Validate an optional generation cap used only for recovery rollouts."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("hard_prompt_recovery.response_length must be an integer")
    if value < 1 or value > max_response_length:
        raise ValueError(
            "hard_prompt_recovery.response_length must be between 1 and "
            f"the rollout response length ({max_response_length})"
        )
    return value


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
