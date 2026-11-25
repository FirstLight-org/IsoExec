"""Router outputs and loss gradients survive capture and stage extraction."""

import pytest
import torch
from torch import nn

import isoexec.nn as inn


class Model(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.embed = inn.Embedding(vocab=32, hidden=16, device=device)
        self.layers = nn.ModuleList(inn.MoE(hidden=16, num_experts=4, top_k=2,
            intermediate=8, shared_intermediate=8, device=device) for _ in range(2))
        self.head = inn.LMHead(vocab=32, hidden=16, device=device)

    def forward(self, input_ids, seq):
        hidden = self.embed(input_ids)
        aux = {}
        for index, layer in enumerate(self.layers):
            value, scores = layer(hidden, seq, return_aux=True)
            hidden = hidden + value
            aux[f"layers.{index}.router.scores_full"] = scores
        return self.head(hidden), aux


def test_component_output_leaves_cannot_alias_each_other():
    class Pair(inn.Component):
        def facts(self):
            return {}

        def _place(self, placement):
            pass

        def forward(self, hidden):
            value = hidden.clone()
            return value, value

    module = Pair()
    with pytest.raises(ValueError, match="another output"):
        module.abstract_output(torch.ones(4, 16, dtype=torch.bfloat16))
