import numpy as np
import pytest

from verl.utils.hard_prompt_recovery import (
    DEFAULT_PROCESS_HINT,
    add_process_hint,
    all_zero_uids,
    source_indices_for_uids,
    validate_recovery_response_length,
)


def test_all_zero_uids_and_source_indices():
    values = {"hard": [0.0, 0.0], "mixed": [0.0, 1.0], "easy": [1.0, 1.0]}
    selected = all_zero_uids(values)
    assert selected == ["hard"]
    assert source_indices_for_uids(np.array(["mixed", "hard", "easy"], dtype=object), selected) == [1]


def test_add_process_hint_copies_messages():
    original = [{"role": "system", "content": "Solve."}, {"role": "user", "content": "Question"}]
    recovered = add_process_hint(original)
    assert DEFAULT_PROCESS_HINT in recovered[0]["content"]
    assert recovered[1] == original[1]
    assert "#### <number>" in recovered[0]["content"]
    assert "write nothing after it" in recovered[0]["content"]
    assert "step by step" not in recovered[0]["content"]
    assert original[0]["content"] == "Solve."


def test_add_process_hint_rejects_decoded_text():
    with pytest.raises(ValueError, match="raw_prompt"):
        add_process_hint("system\nSolve")


def test_validate_recovery_response_length():
    assert validate_recovery_response_length(None, 128) is None
    assert validate_recovery_response_length(64, 128) == 64

    with pytest.raises(ValueError, match="between 1"):
        validate_recovery_response_length(129, 128)

    with pytest.raises(ValueError, match="must be an integer"):
        validate_recovery_response_length(True, 128)
