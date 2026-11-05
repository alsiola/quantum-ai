"""Small GRU and vanilla (tanh) RNN language models, in float64.

Input at step ``t`` is the one-hot previous symbol (a zero vector before the
first symbol); the hidden state starts at zero; a linear readout and softmax
give ``p(y_t | y_<t)``. ``step_log_probs`` and ``predictive_probs`` run the
whole sequence through ``torch.nn.GRU`` / ``torch.nn.RNN`` in one call;
``sample`` steps the same module one symbol at a time (the generation path).
The tests check the fused path against the step-by-step one.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..linalg import REAL_DTYPE
from .base import RecurrentModel, check_sequences, guarded_log

CELLS = {"gru": nn.GRU, "rnn": nn.RNN}


def rnn_parameter_count(cell: str, hidden: int, n_symbols: int) -> int:
    gates = 3 if cell == "gru" else 1
    return gates * hidden * (n_symbols + hidden + 2) + n_symbols * (hidden + 1)


def matched_hidden_size(cell: str, n_symbols: int, budget: int) -> int:
    """Largest hidden size whose parameter count is at most ``budget`` (at least 1)."""
    h = 1
    while rnn_parameter_count(cell, h + 1, n_symbols) <= budget:
        h += 1
    return h


class RNNModel(RecurrentModel):
    def __init__(self, cell: str, hidden: int, n_symbols: int, seed: int = 0):
        super().__init__(n_symbols)
        if cell not in CELLS:
            raise ValueError(f"unknown cell {cell!r}; choose from {sorted(CELLS)}")
        self.cell, self.hidden = cell, hidden
        self.family = cell
        gen = torch.Generator().manual_seed(seed)
        self.rnn = CELLS[cell](n_symbols, hidden, batch_first=True, dtype=REAL_DTYPE)
        self.readout = nn.Linear(hidden, n_symbols, dtype=REAL_DTYPE)
        # PyTorch's default init, but drawn from our generator.
        with torch.no_grad():
            for p in self.rnn.parameters():
                bound = hidden**-0.5
                p.copy_(torch.empty_like(p).uniform_(-bound, bound, generator=gen))
            bound = hidden**-0.5
            self.readout.weight.copy_(
                torch.empty_like(self.readout.weight).uniform_(-bound, bound, generator=gen)
            )
            self.readout.bias.copy_(
                torch.empty_like(self.readout.bias).uniform_(-bound, bound, generator=gen)
            )

    def _inputs(self, seq: torch.Tensor) -> torch.Tensor:
        """``(B, T+1, Y)``: a zero vector, then the one-hot tokens."""
        x = nn.functional.one_hot(seq, self.n_symbols).to(REAL_DTYPE)
        return torch.cat([torch.zeros(seq.shape[0], 1, self.n_symbols, dtype=REAL_DTYPE), x], 1)

    def _log_predictive(self, seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(self._inputs(seq))
        return torch.log_softmax(self.readout(out), -1)  # (B, T+1, Y)

    # -- fused teacher forcing --------------------------------------------------
    def step_log_probs(self, sequences: torch.Tensor) -> torch.Tensor:
        seq = check_sequences(sequences, self.n_symbols)
        logp = self._log_predictive(seq)[:, :-1].gather(-1, seq[..., None]).squeeze(-1)
        guarded_log(logp.exp(), self.diagnostics, self.tiny_prob)  # count only
        return logp

    def predictive_probs(self, sequences: torch.Tensor) -> torch.Tensor:
        seq = check_sequences(sequences, self.n_symbols)
        return self._log_predictive(seq).exp()

    # -- step-by-step interface (generation, and the reference for tests) -------
    def init_state(self, ctx, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(1, batch, self.hidden, dtype=REAL_DTYPE)
        x = torch.zeros(batch, 1, self.n_symbols, dtype=REAL_DTYPE)
        out, h = self.rnn(x, h)
        return out[:, 0], h

    def probs(self, ctx, state) -> torch.Tensor:
        return torch.softmax(self.readout(state[0]), -1)

    def update(self, ctx, state, y: torch.Tensor):
        x = nn.functional.one_hot(y, self.n_symbols).to(REAL_DTYPE)[:, None]
        out, h = self.rnn(x, state[1])
        return out[:, 0], h

    def step_log_probs_stepwise(self, sequences: torch.Tensor) -> torch.Tensor:
        return RecurrentModel.step_log_probs(self, sequences)

    # -- bookkeeping ------------------------------------------------------------
    def state_dim(self) -> int:
        return self.hidden

    def state_kind(self) -> str:
        return "real hidden vector"

    def flops_per_step(self) -> int:
        gates = 3 if self.cell == "gru" else 1
        return gates * self.hidden * (self.n_symbols + self.hidden) + self.hidden * self.n_symbols

    def extra_info(self) -> dict[str, Any]:
        return {"cell": self.cell, "hidden": self.hidden}
