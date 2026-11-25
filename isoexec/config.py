"""Resolved deployment configuration shared by framework adapters."""

from dataclasses import dataclass
from typing import Literal

@dataclass
class IsoExecConfig:
    """Execution choices resolved from a framework's existing configuration."""

    score_temperature: float = 1.0
    """Temperature of reported policy scores; independent of sampling truncation."""

    scoring: str = "contracted"
    """Scoring owner; contracted scoring uses the same binding on both hosts."""

    logprob_bound: float = 0.0
    """Maximum scoring difference; zero requires exact agreement, negative uses the scoring owner’s legacy bound."""

    logprob_gate_interval: int = 1
    """Audit step 1 and every Nth step; skipped audits produce no receipt."""

    full_distribution_comparison: bool = False
    """Opt in to full FP32 vocabulary rows for diagnostics; memory and transfer cost scale with tokens ×
    vocabulary."""

    engine_tp: int = 0
    """Destination tensor-parallel degree for weight sync; zero uses the trainer’s degree."""

    engine_ep: int = 1
    """The engine's expert-parallel degree. Replaces ``ISOEXEC_ENGINE_EP``. The vLLM core_model
    host refuses its own expert parallelism, so anything but 1 is refused at the sync."""

    engine_sp: bool = False
    """Sequence-shard residuals and norms using fixed-tree eager execution; admitted for dense TP2, PP/DP/EP1."""

    enable_prefix_caching: bool = False
    """Reuse admitted prefix cache entries; hybrid hosts validate their state lattice."""
    enable_chunked_prefill: bool = False
    """Allow the engine to split prompts across prefill steps."""
    cudagraph_mode: str = "NONE"
    """Capture mode for the installed eager kernels; compiler-generated numerics are refused."""
    max_model_len: int = 0
    """Engine context limit; zero preserves the inference engine's configured limit."""
    max_num_batched_tokens: int = 0
    """Engine token budget; zero preserves the inference engine's configured budget."""
    trainer_capture_max_tokens: int = 0
    """Resolved upper bound on padded packed trainer tokens; trainer model capture requires a positive bound."""

    model_revision: str | None = None
    """Compatibility field; native adapters leave this ``None`` and use ``tokenizer_path``.
    That snapshot is already resolved to the same commit as the weights."""

    tokenizer_path: str | None = None
    """Configuration/tokenizer directory when weights select a local file."""

    def resolved_scoring(self, enable_isoexec: bool) -> Literal["contracted", "framework_owned"]:
        """``scoring`` with ``auto`` resolved against which model was actually built."""
        if self.scoring == "auto":
            return "contracted" if enable_isoexec else "framework_owned"
        if self.scoring == "contracted":
            return "contracted"
        if self.scoring == "framework_owned":
            return "framework_owned"
        raise ValueError(f"Unsupported scoring owner: {self.scoring!r}")

    def policy(self, enable_isoexec: bool):
        """The :class:`~isoexec.checks.logprob.LogprobPolicy` this deployment declares."""
        from isoexec.checks.logprob import (
            FRAMEWORK_OWNED_MEAN,
            LogprobPolicy,
        )

        scoring = self.resolved_scoring(enable_isoexec)
        bound = self.logprob_bound
        if bound < 0.0:
            bound = 0.0 if scoring == "contracted" else FRAMEWORK_OWNED_MEAN
        return LogprobPolicy(scoring=scoring, bound=float(bound), interval=int(self.logprob_gate_interval))
