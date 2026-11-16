"""Convert audited full trajectories to the DataProto consumed by VERL."""
from __future__ import annotations

from typing import Any

from .contracts import ContractError
from .trajectory import TrainingRow, pack_rows


def as_dataproto(rows: list[TrainingRow], *, max_prompt_length: int, max_response_length: int,
                 pad_token_id: int, device: Any, score_dtype: Any):
    import numpy as np
    import torch
    from tensordict import TensorDict
    from verl import DataProto

    if score_dtype not in {torch.float32, torch.float64, torch.bfloat16}:
        raise ContractError("Explicit floating-point reward dtype is required")
    packed = pack_rows(rows, max_prompt_length=max_prompt_length, max_response_length=max_response_length,
                       pad_token_id=pad_token_id)
    tensors = {}
    identities = {"data_id_list", "rollout_id_list", "policy_version"}
    for key, value in packed.items():
        if key in identities:
            continue
        dtype = score_dtype if key == "token_level_scores" else torch.bool if key == "is_drop_mask" else torch.long
        tensors[key] = torch.tensor(value, dtype=dtype, device=device)
    batch = DataProto(batch=TensorDict(tensors, batch_size=len(rows)))
    batch.non_tensor_batch.update({key: np.array(packed[key], dtype=object) for key in identities})
    return batch
