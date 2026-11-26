# Copyright 2026 the HuggingFace Team. All rights reserved.
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
# Test oracle extracted without changes to equations from Transformers revision
# a005fc82babfe8871d87746decad2dbee100a125/modeling_glm5_next.py.
# The optional kernel decorator is omitted to force the pinned PyTorch equations.
import torch
from torch.nn import functional as F


def l2norm(x, dim=-1, eps=1e-6):
    return x / torch.sqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def chunk_kimi_delta_attention(query, key, value, g, beta, chunk_size=64, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs):
    initial_dtype = query.dtype
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)]
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-06)
        key = l2norm(key, dim=-1, eps=1e-06)
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / query.shape[-1] ** 0.5
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    total_sequence_length = sequence_length + pad_size
    query = F.pad(query, (0, 0, 0, pad_size)) * scale
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    g = F.pad(g, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, g, k_beta, v_beta = [x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, g, k_beta, v_beta)]
    beta = beta.reshape(beta.shape[0], beta.shape[1], -1, chunk_size)
    g = g.cumsum(dim=-2)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    strict_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)
    decay_mask = (g.unsqueeze(-2) - g.unsqueeze(-3)).masked_fill(strict_mask[..., None], float('-inf')).exp().float()
    attn = -(k_beta.unsqueeze(-2) * key.unsqueeze(-3) * decay_mask).sum(dim=-1).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp())
    last_recurrent_state = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device) if initial_state is None else initial_state.to(value)
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)
    for i in range(total_sequence_length // chunk_size):
        q_i = query[:, :, i]
        k_i = key[:, :, i]
        v_i = value[:, :, i]
        g_i = g[:, :, i]
        attn_inter = q_i * g_i.exp() @ last_recurrent_state
        attn_intra = (q_i.unsqueeze(-2) * k_i.unsqueeze(-3) * decay_mask[:, :, i]).sum(dim=-1).masked_fill(mask, 0)
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime
        core_attn_out[:, :, i] = attn_inter + attn_intra @ v_new
        last_recurrent_state = last_recurrent_state * g_i[:, :, -1].exp().unsqueeze(-1) + (k_i * (g_i[:, :, -1:] - g_i).exp()).transpose(-1, -2) @ v_new
    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return (core_attn_out, last_recurrent_state)
