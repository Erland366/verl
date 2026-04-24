# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from verl.utils import attention_utils


def test_attention_utils_fallback_without_flash_attn(monkeypatch):
    real_find_spec = attention_utils.importlib.util.find_spec

    def find_spec_without_flash_attn(name):
        if name.startswith("flash_attn"):
            return None
        return real_find_spec(name)

    monkeypatch.setattr(attention_utils.importlib.util, "find_spec", find_spec_without_flash_attn)
    monkeypatch.setattr(attention_utils, "_index_first_axis", None)
    monkeypatch.setattr(attention_utils, "_pad_input", None)
    monkeypatch.setattr(attention_utils, "_rearrange", None)
    monkeypatch.setattr(attention_utils, "_unpad_input", None)

    hidden_states = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    attention_mask = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=torch.int64)

    unpadded, indices, cu_seqlens, max_seqlen, _ = attention_utils.unpad_input(hidden_states, attention_mask)

    expected = torch.stack(
        [
            hidden_states[0, 0],
            hidden_states[0, 1],
            hidden_states[1, 0],
            hidden_states[1, 2],
        ]
    )
    assert torch.equal(unpadded, expected)
    assert torch.equal(indices, torch.tensor([0, 1, 4, 6], dtype=indices.dtype))
    assert torch.equal(cu_seqlens.cpu(), torch.tensor([0, 2, 4], dtype=cu_seqlens.dtype))
    assert max_seqlen == 2

    repadded = attention_utils.pad_input(unpadded, indices, batch=2, seqlen=4)
    expected_repadded = torch.zeros_like(hidden_states)
    expected_repadded[attention_mask.bool()] = expected
    assert torch.equal(repadded, expected_repadded)

    first_axis = hidden_states.reshape(-1, hidden_states.shape[-1])
    assert torch.equal(attention_utils.index_first_axis(first_axis, indices), expected)
    assert torch.equal(attention_utils.rearrange(hidden_states, "b s h -> (b s) h"), first_axis)
