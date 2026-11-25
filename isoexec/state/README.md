# State

`isoexec.compiler.contract.sema.build_model_contract()` builds a
`isoexec.compiler.contract.contract.ModelContract` from an `nn.Module`, registered op IDs,
and op-owned inference functions. The state package owns state declarations and
state inference; model-level assembly also collects any registered `op_facts`.

```python
from isoexec.core.registry_build import build_registry
from isoexec.compiler.contract.sema import build_model_contract
from isoexec.state.inference import STATE_SPECS_BY_OP

op_registry = build_registry()
contract = build_model_contract(
    model,
    op_ids=op_registry.ops.keys(),
    inference_registry=STATE_SPECS_BY_OP,
)
```

- `contract.py`: portable `StateSpec`, `StateDecl`, and `StateRef`; it re-exports
  the model-owned `ModelContract` for compatibility.
- `../model/contract.py`: the one canonical `ModelContract` definition.
- `../model/sema.py`: model-level collection and `ModelContract` assembly.
- `../model/inference.py`: the mapping to op-owned facts inference and
  validation.
- `sema.py`: module traversal and state inference dispatch; its old
  `build_model_contract` import remains temporarily available for compatibility.
- `validation.py`: state field and identity/reference checks; its old
  `validate_model_contract` import remains temporarily available for compatibility.
- `inference.py`: the mapping to op-owned state inference rules.

GQA and MLA `op_facts` inference lives in
`isoexec/ops/attention/facts.py`. Values are keyed by `StateRef.module_path`, so
two invocations of the same `op_id` retain their own geometry and final `scale`.
The mapping and its typed values are immutable. Omitting `op_facts_registry`
preserves the existing state-only inputs and behavior. Production GQA/MLA
components construct frozen `GQAMetadata`/`MLAMetadata` values extending the
legacy `Attn`/`MLA` inputs with final `scale`. Their execution and ModelContract
inference read that same stored value.

GDN facts inference records the aggregate component's model-owned head geometry
and `conv_kernel`. Kernel-selected chunk size and execution ownership are not
stored in `ModelContract`.

`isoexec.runtimes.vllm.projection` converts GQA/MLA facts into the engine's
TP-local semantic constructor kwargs. They accept no live model or module
metadata. Shared `isoexec.runtimes.projection.project_gdn_state()` validates one
GDN `StateRef`, its `GDNOpFacts`, and its `StateDecl`, then produces the existing
TP-local pool geometry. The production vLLM wrapper consumes these projections.

`isoexec.runtimes.vllm.realization.realize_gdn_state()` and its SGLang peer are
the only wrapper-facing GDN allocation seam. The selected CPR implementation
supplies `chunk_size` there; it is neither authored metadata nor a
`ModelContract` field. The current private continuation buffers remain owned by
the existing pool implementation behind this seam. vLLM's `min_pages` carrier
shape, registered-buffer memory accounting, sleep/wake, and capture behavior
still belong to those concrete runtime layers. They are not a final
`StateHandle` interface.

`isoexec.compiler.contract.binding.bind_stateful_op()` resolves either the exact
`StateRef.module_path` or an explicit lowering-provided native path, verifies
the live module's `stateful_op_id`, and calls its
`bind_stateful_op(op_id, binding)` method. Production GQA/MLA/GDN components retain
the complete `BoundOp` and delegate installation to their existing
`attach_core()` method; generic callable bindings remain supported for
compatibility. Mapping validation requires the intact
native model and exactly one authored/native correspondence per `StateRef`,
resolving every target before slicing, sharding, or realization. It
does not inspect `spec`, infer facts, or derive native paths. Production
GQA/MLA/GDN wrappers use direct same-tree binding; the lowering mapping producer
remains a later migration.

GQA and MLA inference live in `isoexec/ops/attention/state.py`. GQA produces
separate `k` and `v` entries with shape `(num_kv_heads, head_dim)` and `col(dim=0)`.
MLA produces a replicated `kv_latent` entry with shape
`(kv_lora_rank + qk_rope_head_dim,)`. All three entries use logical `bf16`,
`per_token`, `tokens_per_entry=1`, and `retention="full_history"`.

`isoexec/ops/gdn/conv_state.py` derives `conv_q`, `conv_k`, and `conv_v` from the
existing `GDN` metadata, carrying `conv_kernel - 1` inputs in logical bf16.
`isoexec/ops/gdn/state.py` preserves recurrence-only inference for legacy
`schema.GDN`, while production `GDNMetadata` produces one aggregate declaration
containing those convolution tails plus the fp32 `ssm` matrix. These entries are
`per_request` with `col(dim=0)`.

Production `isoexec.nn.GDN` is one stateful module:

```text
GDN: stateful_op_id="gdn.core" -> conv_q / conv_k / conv_v / ssm
```

Its frozen `GDNMetadata` contains model-owned geometry and `conv_kernel`.
Whether the selected core currently records `conv.short` in `BoundOp.owns`
changes execution ownership only; it does not add or remove portable state.
The previous two-child fixture remains supported temporarily so callers can
migrate without changing existing state-only behavior in this PR.

Components declare state with one `stateful_op_id` and canonical metadata in
`spec`. Sema uses that ID for inference and `StateRef.op_id`; an ordinary `op_id`
attribute does not declare state. Metadata is passed unchanged to inference.
Sema derives `state_id` from the module path; the root uses `"<root>"`.
Components do not provide `state_id`, and an explicit value is rejected.
The model layer supplies the established execution-aware module walk, so a
held-but-unexecuted subtree and all its descendants produce no `StateRef`.

`op_ids` comes from the op registry and includes both stateless and stateful ops.
`inference_registry` supplies state inference functions; adding an entry there
does not register an op. An unknown op, or a stateful invocation with missing or
non-callable inference, is an error. An invalid `stateful_op_id` is rejected,
including `None`; it does not make the module stateless.

## Compatibility

Each refactor PR preserves the inputs and call paths accepted by its
`upstream/main` base. If the final interface depends on a later schema or
producer change, the earlier PR accepts the current inputs, the later PR
migrates every producer, and only then may a follow-up remove that compatibility.
A consumer must not require a field before the PR supplying it has landed.

## Invariants

- Each stateful module is inferred once per build.
- `state_id` is `module_path`; the root uses `"<root>"`.
- Equal shapes and module types do not merge state declarations.
- Every reference resolves to one declaration; every declaration has exactly
  one reference in this PR.
- Entry names are unique within a declaration; all state fields are validated.
- Every generated `module_path` must resolve through `model.get_submodule()`
  to the same module instance visited by Sema.
- Module/declaration order is not a binding key.
- Unexecuted subtree roots and descendants produce no state declarations or
  references; duplicate executed paths remain visible for alias refusal.
- A parent may contain multiple stateful children and stateless work. Each
  stateful child has its own module path and reference. Multiple registered
  paths to the same stateful instance, including through an aliased parent,
  are rejected. Stateless module instances may appear at multiple paths.

`isoexec.compiler.contract.validation.validate_model_contract()` checks all materialized
fields and reference links. State inference additionally checks the live module
tree and each invocation's inferred requirements before assembly returns that
contract.

Sema does not execute or trace `forward()`. Exactly-once stateful invocation per
root forward remains a runtime requirement: static traversal cannot detect
repeated or skipped calls. Host projection, realization, binding, and recovery
after possible state mutation are outside this builder.

## Integration responsibilities

- Op registration supplies the registered `op_ids` and the inference callable
  for each supported stateful op. The current mapping can be replaced by another
  `Mapping` without changing Sema or the op-owned inference functions.
- GQA/MLA/GDN component construction supplies `stateful_op_id` and canonical `spec`.
  Model authors using those components do not provide `StateSpec` or perform
  inference registration.
- State inference owns state requirements and reference validation. Model Sema
  owns assembly of the model-owned contract. Parameters, aliases, other model
  fields, and execution identity remain separate integration work.
- Host integration consumes the validated contract, selected implementation
  requirements, and host config. The final `StateHandle`, KV sharing, and
  repeated/skipped invocation enforcement are follow-ups. vLLM's pre-instance
  Mamba shape/dtype queries and SGLang's pre-construction install checks still
  use the legacy config because no model instance or `ModelContract` exists at
  those call points.

GQA/MLA/GDN authoring metadata and component-level BoundOp installation are
integrated into `isoexec.nn`; vLLM and SGLang use ModelContract projection,
runtime realization, and shared binding for all three stateful cores. The CPU
fixtures continue to cover the legacy two-child GDN representation during the
rolling migration.
`kv_sharing_target_layer_name`, target resolution, multiple `StateRef` objects
for one `StateDecl`, and KV-sharing execution remain outside this implementation.

## CPU verification

`tests/test_integration_cpu.py` builds one contract containing GQA, MLA, short
convolution, and GDN recurrence, alongside a stateless projection. Its parameters
are on `meta` or contain CPU NaNs, and its `forward()` raises if called. A second
case adds a registered test-only op with opaque metadata through a read-only
inference mapping and verifies that the existing declarations/references stay
unchanged.

`model/tests/test_op_facts_cpu.py` builds a contract from the complete shipped
Qwen test model and verifies that decoder GDN/GQA rows are present while its
held MTP subtree is absent.

The vLLM and SGLang `test_nn_engine_tier_cpu.py` suites build GQA, MLA, and GDN
contracts before placement, call each production wrapper's `_bind()` loop, and
verify that every per-ref projection reaches the native constructor boundary
and that the resulting `BoundOp` is installed through `bind_stateful_op()`.
The unavailable native attention/state objects and unrelated post-bind checks
are stood in; native runtime execution remains outside these CPU tests.

These checks establish production authoring, contract collection, projection,
and wrapper wiring. They do not establish native runtime execution, GPU parity,
or state round-trip equality.

Run from the repository root:

```bash
pytest isoexec/compiler/tests isoexec/state/tests \
  isoexec/ops/attention/tests/test_op_facts_cpu.py \
  isoexec/ops/attention/tests/test_state_inference_cpu.py \
  isoexec/ops/gdn/tests/test_conv_state_inference_cpu.py \
  isoexec/ops/gdn/tests/test_state_inference_cpu.py \
  isoexec/runtimes/tests/test_gdn_projection_cpu.py \
  isoexec/runtimes/vllm/tests/test_projection_cpu.py -q
pytest isoexec/runtimes/vllm/tests/test_nn_engine_tier_cpu.py \
  -k production_bind -q
ruff check isoexec/compiler/contract isoexec/state \
  isoexec/ops/attention/facts.py \
  isoexec/ops/attention/tests/test_op_facts_cpu.py \
  isoexec/ops/attention/state.py \
  isoexec/ops/attention/tests/test_state_inference_cpu.py \
  isoexec/ops/gdn/conv_state.py \
  isoexec/ops/gdn/tests/test_conv_state_inference_cpu.py \
  isoexec/ops/gdn/state.py \
  isoexec/ops/gdn/tests/test_state_inference_cpu.py
ruff format --check isoexec/compiler/contract isoexec/state \
  isoexec/ops/attention/facts.py \
  isoexec/ops/attention/tests/test_op_facts_cpu.py \
  isoexec/ops/attention/state.py \
  isoexec/ops/attention/tests/test_state_inference_cpu.py \
  isoexec/ops/gdn/conv_state.py \
  isoexec/ops/gdn/tests/test_conv_state_inference_cpu.py \
  isoexec/ops/gdn/state.py \
  isoexec/ops/gdn/tests/test_state_inference_cpu.py
```

For the macOS CPU environment used here, without resolving the project's
CUDA-only dependencies:

```bash
uv run --no-project --python 3.12 \
  --with torch==2.11.0 --with pytest --with numpy \
  python -m pytest isoexec/compiler/tests isoexec/state/tests \
  isoexec/ops/attention/tests/test_op_facts_cpu.py \
  isoexec/ops/attention/tests/test_state_inference_cpu.py \
  isoexec/ops/gdn/tests/test_conv_state_inference_cpu.py \
  isoexec/ops/gdn/tests/test_state_inference_cpu.py \
  isoexec/runtimes/tests/test_gdn_projection_cpu.py -q
uv run --no-project --python 3.12 \
  --with torch==2.11.0 --with pytest --with numpy \
  python -m pytest isoexec/runtimes/vllm/tests/test_nn_engine_tier_cpu.py \
  -k production_bind -q
```
