# IsoExec model compiler design

Models use HF `config.json` and an architecture-named `model.py`, with a
`weight.py` mapping and shard-aware loading. See [How to support new model?](../../docs/support-new-model.md).

This document collects the model architecture, original eight-phase implementation
plan, source/capture/partition/storage/runtime/synchronization contracts, SkyRL
integration and checkpoint recipes. The author-facing package is documented in
the [model support guide](../../docs/support-new-model.md). Internal
[Component extension contracts](#capture-component-registration),
[pipeline diagnostics and failure handling](#partition-diagnostics-and-failure-handling),
and [placement and operation binding](#preparation-component-binding) live here.

The execution path is authored `nn.Module` → `torch.export` and PyTorch
functionalization → IsoExec execution graph and partitioner → executable stage
modules → native Megatron/vLLM adapters. Export was selected because activation
`+=` must preserve later alias/view reads. Components and OpTable remain the
numerical owners; native runtimes retain layer assignment, scheduling,
communication and cache/storage ownership. Authors provide an HF `config.json`,
architecture-named `model.py`, and `weight.py`; see
[the model support guide](../../docs/support-new-model.md). Ambiguous logical-layer
mapping requires an optional model method, not another schema or stage-definition file.

## Phase map

| Phase | Design responsibility |
|---|---|
| [1: Source](#phase-1-source) | HF configuration and JSON source descriptors |
| [2: Capture](#phase-2-export-capture) | Export, functionalization, abstract component outputs and input guards |
| [3: Partition](#phase-3-stage-partitioning) | Native logical-layer assignment, graph cuts and executable stage I/O |
| [4: Preparation](#phase-4-storage-and-placement) | Placement, generic sharding, local storage and checkpoint materialization |
| [5: Runtime](#phase-5-native-runtime-adapters) | Layout analysis/lowering and real Megatron/vLLM stage execution |
| [6: Synchronization](#phase-6-ownership-and-synchronization) | Canonical ownership, gradients, optimizer and weight-update lifecycle |
| [7: Integration](#phase-7-skyrl-and-model-package-integration) | SkyRL source/placement configuration and built-in model packages |
| [8: Qualification](#phase-8-native-qualification-and-limits) | Real W0/update/W1 lifecycle, negative controls and independent evidence |

Dated source/CPU-only milestone descriptions remain historical; the qualification
section below defines the completed native scope. The superseded alternatives and
the frontier-model survey are in this file's git history.


<a id="phase-1-source"></a>

## Phase 1: source


This implementation provides an importable model source, explicit construction,
local `torch.export` capture, logical-layer mapping and executable stage
extraction from native assignments, generic placement and HF/local checkpoint
loading. Phase 5 connects generated programs to Megatron and vLLM adapters.
See [stage extraction](#phase-3-stage-partitioning), [storage preparation](#phase-4-storage-and-placement) and
[native dataflow adapters](#phase-5-native-runtime-adapters). The completed native scope is recorded in [Phase 8](#phase-8-native-qualification-and-limits).

Phase 5 cleanup separates backend-neutral provenance, rule-based pure analysis,
an immutable LoweringPlan and mechanical rewriting from runtime admission.
Native pre-instance queries share bounded immutable facts; executable capture
remains worker-local. Packing lifecycle uses existing architecture facades and
initial admission occurs once after host setup. The model-definition API and original eight phases are unchanged by runtime setup.

Export plus PyTorch functionalization replaces symbolic FX **because activation
`+=` and subsequent alias/view reads must preserve their semantics**. Original
components and OpTable calls remain the execution owners; there are no
model-family mutation rewrites or symbolic-trace fallback. FX GraphModules are
internal executable artifacts. Phase 1 JSON source transport stays unchanged.

Component output metadata follows existing forward composition with OP-owned
abstract bindings. MoE returns the hidden-state input dtype after the complete
combine once capture prepares its output contract; an unprepared component keeps
its FP32 binding result. See [capture](#phase-2-export-capture) for the
preparation contract and native qualification limits.

The original eight-phase plan remains in effect, with one whole phase per review.
See [graph capture](#phase-2-export-capture) for the
actual example-input API, supported envelope and diagnostics, and
[pipeline_layers()](#architecture-automatic-inference-with-an-annotation-only-when-needed) for the
optional author annotation.


<a id="source-package-layout"></a>

### Package layout

| Location | Responsibility |
|---|---|
| `isoexec` | Lazily exports `IsoExecConfig`, `ModelSource` and `ModelSourceError` |
| `isoexec.models` | `ModelSource`, `ModelSourceError` and `build_component_model(source, hf_config)` |
| `isoexec.models.<name>` | Built-in architecture packages `qwen3_5`, `qwen3_5_moe`, `qwen2_moe`, `qwen3` and `qwen3_moe`, each a `model.py` and `weight.py` |
| `isoexec.compiler` | Shared source, capture, partition and preparation machinery, without model-family rules |
| `isoexec.runtimes.megatron` | Megatron trainer construction (`factory.py`) and stage execution (`model.py`) |
| `isoexec.runtimes.vllm` | vLLM engine registration (`registration.register_native`), construction (`factory.py`) and execution (`model.py`) |
| `isoexec.integrations.skyrl` | SkyRL source/placement configuration, checkpoint resolution and worker wiring |

`isoexec.models` exports `ModelSource` for convenience; it is the same class as
`isoexec.compiler.source.ModelSource`. Importing it does not import any built-in
model package.


<a id="source-three-author-files"></a>

### HF configuration and model packages

| File | Construction contract |
|---|---|
| `config.json` | HF `model_type`, `architectures`, dimensions and layer configuration |
| `model.py` | An architecture-named `torch.nn.Module` whose constructor receives the HF config |
| `weight.py` | `checkpoint_conversion_mapping(config)` returns HF checkpoint-to-model conversions |

Keep each checkpoint's config separate from its shared architecture package.
The package may be an ordinary or namespace Python package and must be importable
on every worker. See the [model support guide](../../docs/support-new-model.md)
and the built-in `qwen3_5` and `qwen2_moe` implementations.

`weight_module()` resolves the module when explicitly requested; it does not
load tensors. `prepare_model()` validates the mapping against the whole inventory;
`materialize_stage()` loads local slices. A PP model also needs logical decoder layers.


<a id="source-construction-and-execution-boundaries"></a>

### Construction and execution boundaries

```python
import json
from pathlib import Path

from isoexec.models import ModelSource

checkpoint = Path("path/to/your-qwen-checkpoint")  # a local Hugging Face checkpoint directory
source = ModelSource("isoexec.models.qwen3_5")
raw_config = json.loads((checkpoint / "config.json").read_text())
model = source.build_meta(raw_config)
```

The shared construction entry point also accepts this descriptor:

```python
from isoexec.models import build_component_model

model = build_component_model(source, raw_config)
```

`build_component_model()` requires a `ModelSource`; a bare package string or
checkpoint path is refused.

Native construction is owned by the runtime adapters. Megatron trainer programs
are built by `isoexec.runtimes.megatron.factory`; vLLM engines are configured by
`isoexec.runtimes.vllm.registration.register_native(source, raw_config,
checkpoint=..., engine_kwargs=...)`, which returns native engine kwargs carrying
the JSON source/configuration descriptor. The SkyRL integration drives both.
Meta construction is not executable model admission; the example's component
operations are unbound.


<a id="source-worker-reconstruction-and-identity"></a>

### Worker reconstruction and identity

The source crosses worker boundaries as a plain dictionary with exactly
`package`, encoded using JSON when a byte/text transport is
needed. Use `to_dict()` and `from_dict()` instead of serializing the Python object.
The raw model config travels separately as JSON-compatible data; parsing receives
a private copy. The constructor runs with
PyTorch's default device set to `meta`; registered parameters and buffers must
remain on meta. Import errors inside author code retain their original cause.

```python
import json
from isoexec.models import ModelSource

# Sender: only data crosses the worker boundary.
source = ModelSource("isoexec.models.qwen3_5")
message = json.dumps({
    "source": source.to_dict(),
    "raw_config": raw_config,
})

# Worker: validate the descriptor, then reconstruct from the installed package.
data = json.loads(message)
source = ModelSource.from_dict(data["source"])
model = source.build_meta(data["raw_config"])
```

`from_dict()` requires exactly the `package` field, rejects missing/unknown fields,
and validates the package name without importing it. Source transport does not
use `pickle`/`cloudpickle`. A framework's own RPC implementation remains outside
this descriptor contract.

| Input | Meaning |
|---|---|
| `source.package` | Importable Python package identifying the authored definition |
| `raw_config` | HF JSON parsed by `AutoConfig` using `model_type` |

Workers reconstruct the definition locally. The descriptor does not carry a live
`nn.Module`, CUDA tensors, checkpoint bytes, framework configuration, or a closure.
The caller is responsible for making the same package/configuration available in
each worker environment. Package names alone do not attest equal installed code;
[Phase 6](#phase-6-ownership-and-synchronization) specifies source/configuration/dataflow
digests and checkpoint identity.


<a id="source-actionable-errors"></a>

### Actionable errors

| Failure | Required action |
|---|---|
| Invalid package name or `__main__` | Use a dotted package that can be imported in a fresh worker |
| Invalid source fields | Send exactly `package` via `to_dict()` |
| Missing package or role module | Make `model.py` and `weight.py` available on the worker's Python path |
| Missing `model_type`, `architectures`, or matching class | Supply HF config fields and the named `nn.Module` in `model.py` |
| Constructor returns a non-module | Return the authored `torch.nn.Module` |
| A registered tensor is not on `meta` | Remove explicit device allocation from construction; the error names the parameter/buffer |
| Missing dependency inside author code | Install or fix that dependency; the original `ModuleNotFoundError` is preserved |
| Missing `raw_config` or checkpoint | Supply raw model data and an HF/local safetensors checkpoint |
| Unsupported host, root or native boundary | Follow the named adapter/input/layout diagnostic |

Package modules and configuration parsing must not initialize CUDA or install a
framework runtime. Constructor validation can reject materialized registered
tensors; it is not a sandbox for arbitrary Python side effects in author code.

Source construction is not model admission. Phase 2 capture collects
Inventory/ModelContract and validates dataflow and layer mapping separately;
Phases 3/4 provide stage extraction and storage preparation; Phase 5 provides
the [native execution/input contract](#phase-5-native-runtime-adapters). Distributed source/precision
fingerprints and weight-version acknowledgement are described in Phase 6;
completed GPU evidence and its limits are recorded in Phase 8.
The returned object is the author's module, not a replacement model or native plan.


<a id="phase-2-export-capture"></a>

## Phase 2: export capture and abstract contracts


Component output metadata is now inferred by `Component.abstract_output()` through
its existing forward and operation-owned abstract bindings. MoE returns the
original hidden-state input dtype; LMHead retains FP32 logits. This corrects the
former component-type output-spec table. Stage extraction is described in
[partitioning](#phase-3-stage-partitioning).

Phase 2 uses `torch.export.export(..., strict=False)` followed by
`ExportedProgram.run_decompositions(decomp_table={})`. **The reason for replacing
symbolic FX is support for activation `+=`: later alias/view reads must observe
the mutation.** PyTorch owns functionalization; IsoExec does not rewrite `+=`
manually or introduce model-family exceptions. FX `GraphModule` remains the
executable representation, with no symbolic-trace fallback.

This is local capture and mapping, not stage extraction or native PP admission.
The original eight-phase plan is unchanged. Phase 3 now splits the normalized
program locally; later phases connect native runtimes and qualify PP2/PP4.

Construction and capture pin dtype-less floating factories to `torch.float32`.
Explicit component, parameter and activation dtypes still come from the authored
configuration and precision contract; this does not convert BF16 model weights
to FP32. The caller's previous default is restored on success and failure.
Preinstance capture keys use this same canonical default, so a native worker's
ambient BF16 default does not create a different model definition or constant.

An ATen forward graph cannot preserve arbitrary backward-only author effects.
Capture explicitly rejects custom `autograd.Function`, tensor backward hooks,
and authored grad/inference/autocast mode regions, including `@torch.no_grad()`
helpers. Those effects need a separate supported contract; they must not silently
train with the forward decomposition's backward. Explicit `detach()` remains a
captured tensor operation. Existing opaque Components retain their actual
runtime implementations and backward behavior. Calling capture itself under
`torch.no_grad()` remains supported; that enclosing preparation context is
distinct from a region changed by the authored forward.

Authored ATen operations tagged `torch.Tag.nondeterministic_seeded` are refused
in the raw exported graph, **before** `run_decompositions()` or dead-code
elimination. This includes used and discarded random draws, Bernoulli sampling
and active dropout. Discarding a draw still changes later RNG values; preserving
tensor dependencies alone does not preserve that observable effect. The error
is `PP_CAPTURE_UNSUPPORTED` and names the operation, source location and action.
IsoExec needs an explicit RNG-effect contract for native scheduling, microbatches and
recomputation before admitting such authored operations. A `pipeline_layers()`
annotation cannot repair this. Randomness inside an admitted opaque Component
remains owned by that Component's runtime contract.


<a id="capture-capture-api"></a>

### Capture API

```python
from isoexec.models import ModelSource
from isoexec.compiler.capture.capture import capture

model = ModelSource("my_models.decoder").build_meta(raw_config)
captured = capture(model, (example_input_ids,), seq=example_seq)
print(captured.program.code)
print([(layer.index, layer.path) for layer in captured.layers])
```

The internal preparation API requires example inputs. The integration owns
these examples and their shape policy; authors provide only `config.json`,
`model.py`, and `weight.py`. Optional `Model.pipeline_layers()` remains a method
on the model, not a fourth file. For an authored `forward(input_ids)`, the
separate `seq=` input belongs to the generated program. Capture supplies it to
registered components that declare a sequence-context dependency, while invoking
the authored Model and Blocks with only their own arguments. Generated graph
inputs, input guards, layouts and runtime component calls still include `seq`.
The context comes from the current trace input and is released after that trace;
there is no runtime ambient context or retained example tensor.

Existing authored forwards that explicitly accept `seq` keep their calling
convention:

```python
captured = capture(model, (example_ids,), {"seq": example_seq})
```

Those forwards keep responsibility for passing `seq` to their components;
automatic injection applies only when capture supplies an internal sequence
input. Supplying both an authored `seq` and an internal `seq=` is rejected.

`CaptureInputs.seq` carries the internal context from `text_inputs()` through
train/eval capture and vLLM pre-instance capture. `dynamic_shapes` describes the
complete canonical input tuple, with the internal `seq` last. A captured
model uses the complete program signature. An authored `forward(input_ids)`
produces `built(input_ids, seq)`; additional defaulted, keyword-only or variadic
author parameters make the extra context keyword-only (`built(..., seq=seq)`).

The standard text provider's `max_tokens` is an explicit capture bound, not a
runtime truncation limit. Inputs beyond the captured shape/value constraints
fail before component execution with their actual shapes, recorded ranges and
an instruction to rebuild with a sufficient bound. Framework adapters must
resolve that bound from their real packed-input limits instead of silently
reusing the provider's default for a larger production workload.

`CaptureInputs.root_layouts` is native-provider metadata, not another author file.
Each `RootTensorLayout` names a root pytree tensor leaf and declares global
metadata or CP-local tokens with their axis. The trainer text provider declares
IDs and positions local, with positions on their actual last dimension; packed
offsets remain global. The engine profile retains its own input ownership.
Declarations follow exact Tensor identity aliases already enforced by export's
input guard, so `input_ids` and its `seq.input_ids` alias cannot disagree.
Conflicting aliases or invalid paths/axes refuse capture. The program retains
only paths/layouts, not real capture tensors. Generic CP execution with missing
tensor-leaf declarations reports the missing paths and asks its native provider
to populate `CaptureInputs.root_layouts`.

Plain configuration state may include `SimpleNamespace`, `functools.partial`,
`range`, and `collections.defaultdict`; capture copies their fields, arguments,
factory and ordered entries while preserving internal aliases. This allowance
does not authorize arbitrary objects, generators, callback effects, or mutation
of persistent author state during forward.

The accepted standard text-only entrypoint uses `torch.int64` root token IDs.
Its production adapter must build capture examples and normalize real integer
token IDs using that same specification, rejecting floating-point token IDs.
This is scoped to the text-model adapter; generic capture does not coerce inputs,
and hidden states/other fields keep their own dtypes. Current tests use int64;
the production input provider is still pending runtime integration.


<a id="capture-default-text-input-dtype"></a>

#### Default text input dtype

The standard text-model root parameter `input_ids` accepts int32 or int64 by
default, with no model declaration or policy import. Internal entry normalization
converts int32 to canonical int64 before both export and execution; int64 tensors
are reused. The authored forward consistently sees int64, so dtype-dependent
branches cannot silently disagree with the captured program. Exact dtype guards
remain in place.

Other parameters and intermediate activations are not converted. A generic model
without an `input_ids` parameter keeps its existing exact-dtype input contract.
The helper follows the declared text-entry convention, not a model-family name,
first-argument heuristic or weight dtype. Floating-point/bool/None `input_ids`
are rejected.

Repeated references to the same token Tensor are converted together; caller
inputs and unaffected containers remain unchanged. Int32 inputs with distinct
storage-alias peers fail instead of silently breaking alias relationships.
Whole captured and partitioned programs normalize at their root entry. Direct
StageModule calls take canonical inputs; native adapters must use the same
internal normalization before binding them. Phase 5 now implements the shared
[text-input provider and native root binding](#phase-5-native-runtime-adapters); generic capture still
uses the explicitly supplied example contract.

Shapes are static by default. `dynamic_shapes`, when supplied, is a tuple in
`forward` parameter order, including default and variadic parameters. Entries
follow PyTorch's nested dynamic-shape format. For example, a two-Tensor forward
can share a symbolic leading dimension:

```python
tokens = torch.export.Dim("tokens", min=2, max=128)
captured = capture(model, (hidden, residual),
                   dynamic_shapes=({0: tokens}, {0: tokens}))
```

Export's shape/range/static-value guards remain in the executable program.
Tensor metadata and duplicate-input identity assumptions are checked as well;
an input that violates capture assumptions fails before component execution.
Dynamic tensor inputs currently require canonical contiguous layout. Different
shape policies, dtypes, devices, grad flags, or input alias patterns require a
matching capture, rather than silent reuse. A root `SequenceContext` is a keyed
pytree; its host integers are specialized unless declared dynamic, and its
optional-field structure is specialized. The standard `text_inputs()` provider
declares `max_q` and `max_kv` dynamic. Generic captures that leave these integers
static require a matching capture when their values change, even with dynamic
token shapes.

The model must be unplaced. Meta construction is normal; CPU models are useful
for numerical checks. Capture prepares component-owned output policies, but does
not load weights, place the original model, bind its runtime OpTable, or initialize
a native runtime. Metadata inference uses an isolated copy at logical degree-one
geometry, with a private AbstractOpTable; this is not a deployment topology limit.


<a id="capture-component-boundary-and-execution"></a>

### Component boundary and execution

```text
capture: prepare_output_contract() on original components
    -> Component.abstract_output(): isolated component copy + FakeTensors
    -> existing component.forward() + AbstractOpTable
    -> output shape / dtype / stride / device
    -> opaque export marker + PyTorch functionalization
    -> original component calls + executable output metadata checks
```

`Component.abstract_output()` runs the same component forward on a private copy
with unsharded degree-one geometry. Only that copy receives abstract operation
bindings; the author model remains unplaced and keeps its original parameters,
ties and slots. Pure PyTorch operations reuse their current definitions. Opaque,
value-dependent operation cores have explicit OP-level metadata rules. Unknown
operations fail with a request for their abstract contract. There is no component
or model-family dtype switch in the exporter.

Abstract execution excludes its inner operations from the surrounding export
trace. The trace retains component markers, which have no real numerical kernel.
Normalization replaces every marker with the original component call and OpTable
dispatch. Shape/stride expressions remain live, including padded LMHead outputs
whose trimmed vocabulary has noncontiguous strides. Output checks compare dtype,
shape, device, stride and storage offset and reject input aliases/lazy views.
These checks are not a proof of numerical or distributed equivalence.


<a id="capture-moe-output-policy-and-compatibility"></a>

#### MoE output policy and compatibility

Before capture, the generic preparation walk calls each component's
`prepare_output_contract()`. MoE selects `boundary_out_dtype="input"`; its real
forward records the ORIGINAL hidden-state dtype and casts only the complete
routed/shared result at the final component boundary. The same cast covers the
normal and `combined_local` returns. Router/accumulation precision is unchanged.
Abstract execution observes this real cast, so BF16 inputs produce BF16 metadata
and FP32 inputs produce FP32 metadata where the underlying operations admit them.

An unprepared MoE keeps its FP32 binding result and facts. Preparation is
idempotent and precedes placement. It does not globally alter the shared combine
operation or replace an existing native model. A backend that already rounded to
BF16 cannot recover a wider-precision result by upcasting: native precision and
placement admission must still prove compatibility. This CPU phase does not
qualify new GPU dtypes, native fused MoE kernels or distributed gradients.

For direct component inspection, use:

```python
component.prepare_output_contract()
fake_output = component.abstract_output(example_hidden, example_seq)
print(fake_output.shape, fake_output.dtype, fake_output.stride())
```

Normal model authors do not need these internal calls: `capture()` prepares them.
There is no extra author file or new required model annotation.

The component metadata envelope accepts a Tensor or a pytree of Tensors. Every
output must have strided storage with zero storage offset,
no lazy conjugate/negative flag, no alias to input/parameter/buffer storage, and
no observable argument mutation during abstract execution. Capture outside
`torch.inference_mode()` with ordinary example tensors; `torch.no_grad()` is
supported. This preparation restriction does not prohibit later native inference
execution. Input value-dependent validation remains a runtime/adapter obligation.

The original runtime `SequenceContext` still reaches restored components, so
existing cache identity is preserved. OP-owned RoPE metadata does not execute
runtime angle-cache writes; outer authored reads/writes of those caches remain
refused. MoE's legacy `aux` dictionary writes remain outside this pure capture
contract. Use `return_aux=True` to return `(hidden, scores_full)` explicitly;
both tensors follow the caller's token stream, including under SP. The authored
model can return `(logits, {name: scores})` while retaining its terminal LMHead.
The abstract table describes logical unsharded operation semantics, not the exact
intermediates of every fused runtime implementation.


<a id="capture-component-registration"></a>

### Component extension contracts

Ordinary enclosing `nn.Module` blocks expose their child calls to capture.
A derived `isoexec.nn.Component` is an opaque execution boundary and requires
exact-class registration through `register_capture_component()` at import time
in `model.py`.

Registration does not infer or replace `abstract_output()`, tensor axes/state,
token layout, numerical facts, or OpTable placement. These remain the
Component's contract; inherited methods are valid only when their semantics
still apply. Capture restores the registered component instance. Register at
import time so every worker sees the same definition.

Subclasses require their own registration. Changing a registered class's
`forward()` afterward, hooks, and instance `forward` replacements remain
unsupported. A newly registered opaque component cannot hide other stateful
invocations: compose Attention/GDN and other stateful components in an ordinary
`nn.Module` so capture can validate their individual calls. Unknown types fail
with an authored location and a request for `register_capture_component()`.
No fourth author file or runtime-specific annotation is required.

The registered definition is restored as the actual runtime component.
Registration supplies identity and admission, not an inferred implementation of
the component's numerical or state semantics. Source:
[exact-class registration](capture/component_capture.py).


<a id="capture-artifact-and-mapping"></a>

### Artifact and mapping

| Field | Meaning |
|---|---|
| `program` | Normalized executable `GraphModule`, original components, input guards |
| `calls` | Module invocations, original FQNs/types, parent calls and scoped nodes |
| `layers`, `layer_source` | Ordered logical layers, inferred or explicitly annotated |
| `head` | Original terminal LMHead call, including its actual keyword arguments |
| `inventory` | Existing logical tensors, shard axes and tied aliases |
| `contract`, `state_nodes` | Existing ModelContract and captured stateful call owners |
| `training_modes`, `execution_modes` | Model/ambient settings used for capture |

Automatic inference requires one unambiguous serial `ModuleList` whose members
have unique registered paths, execute once in order, contribute to the head and
cover the computational component calls. Names such as `layers` and `blocks`
have no special meaning. Nested candidate lists or independent branches require
an explicit mapping:

```python
def pipeline_layers(self):
    return self.units
```

The method returns existing modules in logical-layer order. An invalid explicit
mapping never falls back to inference. Strings, overlapping scopes, aliases,
unregistered modules, unused layers, repeated enclosing-layer invocations and
tuple groups are refused. One enclosing module represents one logical layer.
An explicit mapping can describe a fixed DAG; it does not admit the later PP
boundary payload or layout. Inter-layer tensor work remains in the graph.

Each stateful module executes once per root forward. An opaque built-in composite
owns its children's state sites. Weight tying remains allowed. The author returns
one terminal `isoexec.nn.LMHead` output directly, with no post-head operation.

Artifacts remain local to a worker. Send Phase 1 JSON source/configuration fields
and capture locally; do not pickle GraphModules, model objects or callables.


<a id="capture-mutation-and-refusal-boundaries"></a>

### Mutation and refusal boundaries

Supported activation `+=` and named-view mutation are functionalized and checked
against eager forward, input gradients and parameter gradients. This does not
promise all in-place syntax is valid:

- Writes to user-input tensors or persistent parameters/buffers require writeback
  semantics and are refused by exported output-signature validation.
- Some slice assignments produce `aten.copy.default`, whose backward is missing
  in the tested PyTorch 2.11 environment. These are explicitly refused; a valid
  functional expression is required. No homemade decomposition is installed.
- Tensor-dependent Python control flow, unsupported higher-order regions and
  changing grad/inference/autocast context around component calls are refused.
- Hooks, instance `forward` overrides, custom Component implementations without
  export contracts, or executable authored children hidden inside an opaque
  component are refused. Put composition in an ordinary `nn.Module`.
- Configuration branches are fixed at capture. Keep Python configuration/external
  state immutable; do not use runtime object identity/type inspection as dynamic
  dispatch. Capture is not a sandbox for arbitrary Python side effects.

Registration/tensor/mode/scalar/dtype snapshots detect common model mutation during
capture and annotation. Recapture after changing configuration or execution
settings. Phase 3 must retain the entry guards and functional value versions when
extracting stages; capture alone does not prove cross-rank alias or PP semantics.


<a id="capture-diagnostics-and-validation-scope"></a>

### Diagnostics and validation scope

`PipelineError` exposes `code`, `phase`, `model`, `location`, `reason`,
`required_information`, `action`, and `context`; `to_dict()` provides plain error
data. Original causes and authored source locations are retained where available.

| Code | Action |
|---|---|
| `PP_CAPTURE_UNSUPPORTED` | Fix the named unsupported operation, effect or export input contract; layer annotation cannot fix it |
| `PP_LAYERS_REQUIRED` | Add `pipeline_layers()` returning ordered existing modules |
| `PP_LAYERS_INVALID` | Correct the named reference/order/scope conflict |
| `PP_STATEFUL_INVOCATION` | Invoke each stateful module exactly once and remove ambiguous module aliases |

Later phases add assignment/boundary/native admission and worker error propagation.
Runtime input-contract violations fail directly before unplaced kernels execute.

Validation uses a small non-Qwen CPU fixture, exact forward/gradient comparisons,
original component/OpTable identity, tied weights and negative controls. Meta
Attention/GDN capture checks state references, not native kernels or PP transport.
All PR Python files must pass the scoped repository `basedpyright` standard gate.
Historical symbolic-FX receipts remain separate from current export evidence.

Primary sources: [PyTorch export](https://docs.pytorch.org/docs/2.11/export.html),
[custom operator fake implementations](https://docs.pytorch.org/docs/2.11/library.html#torch.library.register_fake),
[capture implementation](capture/capture.py), [component boundary](capture/export_boundary.py),
[component abstract execution](../nn/abstract.py),
[operation metadata bindings](../ops/abstract.py),
[input guards](capture/export_inputs.py), [normalization](capture/export_normalize.py),
[layer mapping](capture/layers.py), [existing model contracts](contract/sema.py).


<a id="phase-3-stage-partitioning"></a>

## Phase 3: stage partitioning


Phase 3 converts a captured logical program and native layer assignments into
local executable stages. It preserves authored operations and gradients; it
does not implement native scheduling, P2P communication, sharding or checkpoint
loading. Those remain later phases of the implementation plan.


<a id="partition-api-and-assignment-ownership"></a>

### API and assignment ownership

```python
from isoexec.compiler.capture.capture import capture
from isoexec.compiler.partition import partition

captured = capture(model, example_args, example_kwargs)
# Example for four logical layers; production adapters supply native assignments.
stages = partition(captured, [(0, 2), (2, 4)])
stage = stages[pp_rank]
```

Assignments are ordered half-open `(start, stop)` ranges, one per stage. They
must cover every logical layer exactly once, without gaps or overlaps. Megatron
currently provides `(offset, count)` via its layer-assignment helpers; its adapter
will pass `(offset, offset + count)`. vLLM provides `(start, stop)` through
`get_pp_indices()`. The partitioner never invents a balanced allocation.
`partition()` returns one `StageModule` per assignment; it does not assemble a
callable whole model. CPU equivalence tests use the test-only
[`compose_stages()`](tests/partition_helpers.py) helper. That helper is not a
native scheduler or production communication path.

Prefix work belongs to the first stage, suffix/final norm/head work to the last.
Work between logical layers remains with the preceding layer in original graph
order. All dependencies are checked for forward stage order. Empty ranges are
represented by transparent stages carrying live values; this is a local graph
capability, not admission of an empty stage by a native backend.

Partition before placement/local parameter sharding. Stage modules retain the
original held components and tensors at original relative FQNs. They are not
replacement native model definitions and do not retain the full source model as
a hidden registered child.


<a id="partition-stage-input-and-output-contract"></a>

### Stage input and output contract

`stage.plan.inputs` gives the positional argument order for the generated stage.
Each `BoundaryValue` records its name, source and available metadata:

| Field | Meaning |
|---|---|
| `source="root"` | Original model argument that the runtime adapter must bind locally |
| `source="activation"` | Computed value carried from the preceding pipeline boundary |
| `kind` | Tensor, scalar, context, or explicitly unknown metadata |
| `shape`, `dtype`, `device` | Logical captured tensor metadata; symbolic dimensions are described as expressions |

Non-final stages return a tuple matching `stage.plan.outputs`; the final full
stage returns the original logits Tensor. Carry every value live across a cut,
including saved residuals and values consumed only several stages later. Empty
middle stages forward those values. No detach, value recomputation or implicit
cross-rank storage sharing is introduced.

For a model returning named auxiliary tensors, `stage.plan.aux_outputs` records
the names and graph nodes owned by that stage. The stage graph returns its normal
result together with that local dictionary. Auxiliary tensors are not inserted
into the hidden-state PP wire. `RuntimeStage.run()` keeps its normal result and
exposes `last_aux`; `return_aux=True` returns the result and dictionary without
retaining it. The Megatron host exposes `last_aux`. Megatron includes
auxiliary tensors in reentrant checkpoint outputs before rebuilding the dictionary,
so recomputation preserves the loss gradient instead of retaining no-grad values.
Loss composition and any cross-stage aggregation remain owned by the trainer.

Standard MoE load balancing is selected by the training configuration, not by a
Qwen-specific loss file. SkyRL users enable it with
`trainer.policy.megatron_config.moe_router_load_balancing_type=aux_loss` and
`trainer.policy.megatron_config.moe_aux_loss_coeff=0.001` (or another explicit
coefficient). `none` or a zero coefficient disables the native objective;
`router_aux_loss_coef` in the HF checkpoint does not enable it automatically.
The IsoExec SkyRL adapter prepares each layer's masked selection counts and full
softmax probabilities, then calls native `switch_load_balancing_loss_func` directly.
It sums the layer losses without pooling router statistics or averaging layers.

For a different objective, an author package can still export `build_aux_loss`
from `model.py`. This is an optional custom helper, not a requirement for Qwen.

`ModelSource.build_aux_loss(raw_config, metadata=..., placement=...,
aux_outputs=...)` resolves that optional builder. It receives a private copy of
the raw configuration, global `ModelMetadata`, local `Placement`, and the names
of that stage's declared auxiliary outputs. The builder validates its required
tensors and supported placement before returning an `AuxiliaryLoss` callable.
An absent builder or a `None` result selects no authored auxiliary loss. Native
load balancing can still be selected by the training settings above. Selecting
both an authored helper and native `aux_loss` is rejected to prevent duplicate
gradients. There is no model-name registry or automatic choice based on
`router_aux_loss_coef`.

The callable receives `(aux, token_mask, cu_seqlens)`: the current differentiable
outputs, a flat packed token mask, and padded sequence boundaries in those same
rows. It returns a weighted scalar loss and scalar metrics. A helper owns its
formula, coefficients, and token/layer/sequence reductions. It must produce a
finite differentiable zero for an entirely masked microbatch. New helpers may
require raw logits or other explicitly authored outputs; the adapter does not
interpret or rename `aux` keys. Adding a helper does not change the model's
forward graph or the trainer's backward/optimizer loop.

The IsoExec SkyRL adapter subclasses the existing wrapper only at
`forward_backward_mini_batch`, keeping SkyRL's PPO/SFT loss implementation
unchanged. Before Ray creates policy actors, configuration resolution selects an
IsoExec worker subclass whose registered `forward_backward` method scopes row
observations around the native worker body. Ray invokes the registered class
method, so installing an instance wrapper after actor creation is insufficient.
Auxiliary training refuses to run without that observation scope.
Within one schedule it pairs each forward with that microbatch's
original mask and padding metadata, using the microbatch counts supplied by SkyRL.
The host consumes the current differentiable `last_aux` after checkpoint outputs
have been restored. Native `MoEAuxLossAutoScaler` attaches the weighted auxiliary
loss to the output while preserving its forward values. The native schedule
scales auxiliary gradients; the adapter corrects for fully padded microbatches
so the objective averages over real microbatches. Recompute does not inject a
second loss. Context and tensor references are cleared on success or failure.

The adapter detaches only logging metrics and appends them to the existing
per-microbatch reports. SkyRL's reported task loss remains its original scalar;
auxiliary loss is reported separately. Forward-only scoring does not inject aux
gradients. No `_aux_loss_fn` field or added loss term is needed in SkyRL's source.

This is a per-layer, per-microbatch objective. With SP, the adapter reduces expert
assignment counts and valid-token counts over the TP group while preserving each
rank's differentiable probability slice. Each PP stage computes its own layers'
losses; the adapter merges stage metrics after the native schedule. DP follows
SkyRL's real-microbatch averaging. Auxiliary loss requires CP=1 and a MoE carrier
with `calculate_per_token_loss=False`. Standard `aux_loss` requires full softmax
probabilities; other native balancing modes and CP need their own statistics and
mask integration. These rules do not establish GPU correctness for untested
parallelism degrees or combinations.
Independently scheduled router-bias updates and their synchronization are outside
this loss interface.

Root arguments are distinct from activation payloads. For example,
`SequenceContext` is supplied by the adapter, not serialized as a KV-cache payload.
A later stage may also require a root Tensor; the adapter must satisfy that
requirement explicitly. Scalars or unknown boundary values must receive native
admission rather than being silently discarded or relabeled as hidden states.

The generated stages use canonical positional values and do not retain a
whole-model root calling convention. The test-only composition helper copies
the captured whole-model pre-hooks so positional/keyword parity and input-guard
refusals remain covered. Production adapters normalize root arguments before
binding stage inputs. Input guards execute once at the first stage; output
metadata checks stay with their corresponding component calls. Native worker
error propagation and remote input validation remain adapter work.

Captured metadata is logical and unsharded. Native TP/SP layouts and selected-row
head execution still require their later placement/adapter contracts. Changing
root dtype, optional-field structure or specialized Python values is not made
valid merely by splitting the graph. The standard text-root int64 decision and
pending production example-input provider remain unchanged.


<a id="partition-terminal-bodyhead-interface"></a>

### Terminal body/head interface

The final stage also exposes:

```python
final_stage = stages[-1]
head_args = final_stage.forward_body(*last_stage_args)
logits = final_stage.compute_logits(*head_args)
```

`head_inputs` names those cut values in order. The tuple can contain more than
one Tensor or scalar; it is not implicitly a single `hidden_states` argument.
The original LMHead call, arguments, keywords and trailing output validation are
preserved. The body contains no head execution, and `compute_logits()` does not
rerun the body. Held `get_attr` tensors stay with their consumer instead of
entering the activation tuple.

`body_program` and `head_program` expose the corresponding GraphModules for
inspection. They are views over the full stage's held objects, not independently
registered parameter owners. Stage methods refresh their held bindings so that
buffer replacement or `load_state_dict(assign=True)` does not leave stale
references. Use the StageModule methods rather than caching a graph view across
storage replacement. The existing stage still owns optimizer/checkpoint storage.

This interface is preparation for vLLM's separate logits call. It does not yet
implement vLLM row selection, local vocabulary geometry or native state binding.


<a id="partition-ownership-metadata"></a>

### Ownership metadata

The captured `ExecutionGraph`, and later `ModelMetadata`, retain the global
`Inventory` and `ModelContract`. Each StagePlan records its logical layers,
local module paths, input/output schemas, tensor bindings and projected state
contract.

- `TensorBinding.local_name -> canonical_name` preserves global identity. A final
  `head.weight` can map to a first-stage `lexicon.weight` even when the canonical
  name is absent from the final stage's module tree.
- Partitioning derives consumer stages from actual stage bindings. Each
  `PlacementEmissionPlan.consumer_stages` retains that result; global tensors
  without executable consumers remain in the global `Inventory` without a
  fabricated emission record.
- `canonical_tensor_ownership()` derives the authored logical owner from the
  global `Inventory`, independently of PP. `emission_plans` then assign
  consumers, per-stage aliases and a physical emitter. A unique held declared
  tie owner takes precedence; otherwise `first_consumer` chooses the emitter
  only for implicit sharing. A named explicit TieDecl owner must be held by
  exactly one consumer stage; missing or ambiguous owners raise an actionable
  error rather than changing policy. Every stage retains the same emission
  plans, including nonconsumer ranks needed by the native synchronization
  schedule.
- Partitioning derives state ownership from each captured stateful invocation
  node and projects the matching declarations/references into each
  `StagePlan.contract`. Opaque DecoderBlock children retain their original state
  IDs and paths. Persistent state shared across stage owners is refused; no
  cache transport is inferred from an activation dependency.
- Synced buffers retain their global declarations. Other held buffers/constants
  remain local attributes, with synchronization policy left explicit.

All StagePlans and GraphModules remain worker-local. Continue sending the Phase 1
JSON source/configuration data and reconstructing locally; no pickle transport
is added.


<a id="partition-errors-and-evidence"></a>

### Errors and evidence

`PP_ASSIGNMENT_INVALID` identifies malformed/gapped/overlapping/out-of-range
assignments. `PP_BOUNDARY_UNSUPPORTED` identifies a backward stage dependency,
missing global tensor/state identity, cross-stage persistent-state sharing,
placement before extraction, or a collision with StageModule adapter attributes.
Normal author names such as `head` remain valid. The reserved API names include
`plan`, `head_inputs`, `body_program`, `head_program`, `forward_body`,
`compute_logits`, `_head_programs`, and `_refresh_view`.

One CPU fixture covers PP2/PP4, an empty middle stage, cross-cut residual/view
mutation, input/parameter gradients, tied padded head, root keyword binding,
input guards, state projection and synced-buffer replacement. Separate local
probes cover dynamic token counts and leading/trailing empty stages. These
checks establish local graph equivalence, not native PP or distributed gradients.

`StagePlan.cross_stage_parameters` projects the completed global parameter
consumer map to each holding stage as immutable canonical-name/stage pairs.
Buffers are excluded. It does not select a gradient or emission owner and does
not make partition reject a representable graph. Native adapters consume the
facts for their current ownership-capability admission.

Primary sources: [partition implementation](partition.py), [terminal head split](head.py),
[Megatron assignment](../runtimes/megatron/model.py),
[vLLM assignment](../runtimes/vllm/model.py),
[PyTorch FX GraphModule](https://docs.pytorch.org/docs/2.11/fx.html#torch.fx.GraphModule).


<a id="partition-diagnostics-and-failure-handling"></a>

### Pipeline diagnostics and failure handling

The [accepted author contract](#architecture-automatic-inference-with-an-annotation-only-when-needed)
records automatic inference and `pipeline_layers()`. This section records
internal partition and diagnostic requirements.

A **module reference** is an existing `nn.Module` object. Its **FQN** is its
fully qualified path in the current model tree, such as `decoder.blocks.0`.
IsoExec resolves references against the model tree and preserves them through
recorded invocations and normalized graph scopes. A rename changes the FQN for
the new build and may require checkpoint-mapping changes; references avoid a
second list of author-maintained string paths.

For example, if the native assignment distributes four layers across two ranks, IsoExec can derive these stages from the actual dataflow:

```text
Stage 0: input_ids → embed → Block 0 → Block 1 → boundary values
Stage 1: boundary values → Block 2 → Block 3 → × 0.5 → norm → head
```

Functional operations such as multiplication and residual additions must be preserved even though they are absent from the annotation. The graph determines which activation values cross a boundary, potentially including multiple tensors. Execution on a selected backend still requires adapter capability and layout validation.

The state contract and runtime bindings retain ownership of persistent KV/GDN caches. A graph boundary does not implicitly turn every cache into a PP payload. The complete authored `forward()` still ends with `LMHead` logits; the adapter handles the body/head separation required by vLLM.

<a id="pipeline-5-required-partition-failure-diagnostics"></a>

#### Diagnostic fields

A failure reaches the caller as an exception. Capture and mapping failures use `PipelineError` from `isoexec.compiler.capture.graph`, including plain diagnostic fields through `to_dict()`. Assignment and backend-boundary failures additionally require the context below. Logging a warning or returning an apparently usable model is insufficient.

Each error must include:

| Field | Information the user needs |
|---|---|
| `code`, `phase` | Failure category and whether it occurred during capture, layer mapping, partitioning, or backend admission |
| `model`, `location` | Model, annotation index, module path/FX node, and authored source location when available |
| `reason` | Observed evidence and why execution cannot proceed, such as two candidate layer hierarchies |
| `required_information` | Information the author must supply; explicitly use `none` when information is not missing |
| `action` | An actionable correction, with a minimal code example when useful |
| `context` | For partition failures: backend, requested PP, layer assignment, failing boundary, and known tensor layouts |


<a id="pipeline-example-1-ambiguous-layer-inference"></a>

##### Example 1: Ambiguous layer inference

```text
[PP_LAYERS_REQUIRED] Cannot uniquely infer PP logical layers.
phase: layer_mapping
model: MyModel
reason: Both outer blocks and their nested blocks are candidate layer units.
candidates:
  outer: [blocks.0, blocks.1]
  inner: [blocks.0.units.0, blocks.0.units.1, blocks.1.units.0, blocks.1.units.1]
required_information: Ordered existing nn.Module instances representing logical decoder layers.
action: Add Model.pipeline_layers() and select the intended layer level.
example: If each outer block is one logical layer, return self.blocks.
docs: isoexec/compiler/DESIGN.md#architecture-automatic-inference-with-an-annotation-only-when-needed
```

This is the case that requires `pipeline_layers()`. Diagnostics may list candidate interpretations, but must not choose the author's intended model grouping by guesswork.


<a id="pipeline-example-2-invalid-annotation-format"></a>

##### Example 2: Invalid annotation format

```text
[PP_LAYERS_INVALID] pipeline_layers()[1] is not an nn.Module.
phase: layer_mapping
model: MyModel
location: Model.pipeline_layers()[1]
observed: str "blocks.1"
required_information: The existing module instance for logical layer 1.
action: Return self.blocks[1], not "blocks.1".
```


<a id="pipeline-example-3-complete-information-but-an-unsupported-backend-boundary"></a>

##### Example 3: Complete information, but an unsupported backend boundary

```text
[PP_BOUNDARY_UNSUPPORTED] The selected adapter cannot execute the required stage I/O.
phase: backend_admission
context: Report the actual backend, PP degree, source stage and destination stage here.
boundary: blocks.1 -> blocks.2
required_values: [hidden_states, saved_residual]
reason: Report the concrete missing transport/layout capability, with expected and actual specs.
required_information: none
action: Use a supported partition/backend configuration, or implement the missing adapter capability.
```

This is a diagnostic template, not a claim that vLLM cannot transport multiple tensors. The implementation must identify the capability actually missing from the selected adapter. Adding `pipeline_layers()` does not repair this failure; a framework capability gap must not be presented as missing author metadata.


<a id="pipeline-6-match-the-correction-to-the-failure"></a>

#### Required corrections

| Code | Cause | Required information or correction |
|---|---|---|
| `PP_CAPTURE_UNSUPPORTED` | Fixed dataflow cannot be captured, for example because of a tensor-dependent Python branch | Identify the source/operation and require supported fixed dataflow or an existing component; a layer annotation cannot repair capture |
| `PP_LAYERS_REQUIRED` | No unique layer mapping and no explicit annotation | Supply ordered existing module references through `Model.pipeline_layers()` |
| `PP_LAYERS_INVALID` | Invalid element type, newly constructed/unregistered module, ambiguous alias, overlapping scopes, incorrect order, or an unexecuted declared module | Identify the annotation index, paths, and observed invocation order; correct the mapping/registration instead of reordering the graph to fit an invalid annotation |
| `PP_STATEFUL_INVOCATION` | A stateful module is invoked repeatedly within one root-forward dataflow | Report its path and every invocation site; repeated annotation entries do not create independent state. Intentional state sharing or repeated invocation requires a separate semantic extension |
| `PP_ASSIGNMENT_INVALID` | Native assignment is incompatible with the layer count or requested PP | Report the layer count, PP configuration, and native constraint; correct deployment configuration instead of inventing extra layers |
| `PP_BOUNDARY_UNSUPPORTED` | The required values are known, but the adapter cannot implement their transport, layout, or state dependency | Report producers/consumers, values, shapes/dtypes/layouts, and missing capability. If component metadata is missing, identify the component and field; do not indiscriminately request author-written stage I/O |

The stateful-invocation rule concerns one authored root-forward dataflow. It is distinct from successive decode forwards or activation-checkpoint recomputation. Weight tying is also distinct from reusing state; diagnostics must preserve these distinctions.

Export constraints and unsupported Python control flow still need validation. Supplying a layer list does not repair capture, missing mutation semantics, or unsupported runtime state effects. In-place syntax alone is not a refusal: the selected frontend must functionalize supported activation mutations and preserve the affected reads. [Export requirements](https://docs.pytorch.org/docs/2.11/user_guide/torch_compiler/export.html)


<a id="pipeline-7-failure-handling-requirements"></a>

#### Worker failure handling

1. Report failures discoverable during capture, mapping, or partitioning before actual execution. Checks requiring native worker settings must report failures to the driver/user during worker initialization.
2. Propagate worker-initialization failures through the framework's error mechanism. Do not swallow exceptions and leave other ranks waiting for a stage that will never run.
3. Preserve the original cause and relevant authored location when wrapping an exception. A generic "build failed" message is insufficient.
4. Never silently switch to PP=1, ignore an explicit annotation, discard boundary values, or substitute a different native model.
5. Capture, validate, and partition again after a correction. Changes to modules, dataflow, or deployment configuration invalidate an unverified previous StagePlan.

Primary sources: [logical-layer validation](capture/layers.py),
[error representation](capture/graph.py), [partitioning](partition.py),
[model contracts](contract/sema.py), [state invariants](../state/README.md#invariants).


<a id="phase-4-storage-and-placement"></a>

## Phase 4: placement and storage preparation


Phase 4 prepares a stage's local storage from the same global model facts used
for capture. It reuses existing tensor slicing, component placement, OP binding
and packing implementations. It does not replace native scheduling or implement
new distributed semantics. The Phase 5 Megatron/vLLM execution contract is
documented in [native runtime adapters](#phase-5-native-runtime-adapters).


<a id="preparation-author-checkpoint-contract"></a>

### Author checkpoint contract

The author's `weight.py` defines `checkpoint_conversion_mapping(config)` and
returns Hugging Face `WeightRenaming` and `WeightConverter` objects. `config` is
parsed from the same effective HF JSON used to construct the complete model.
There is no fallback to the former author API. See
[checkpoint conversions](#checkpoint-conversion-mapping) and the
[model support guide](../../docs/support-new-model.md) for the API.

IsoExec checks global canonical coverage, ties, source encodings and shapes,
then copies only the current PP/EP/TP rank's held shards. A failed conversion,
copy, or packed-storage refresh leaves the stage blocked until a complete
successful reload. The native owner must quiesce execution before loading.

Both local/private safetensors paths and pinned HF repository snapshots use
this loader. No upstream model instance is constructed during production loading.


<a id="preparation-component-binding"></a>

### Component placement and operation binding

The preparation path applies this component setup contract to the extracted stage's original components.

A unified model constructs `Attention`, `GDN`, `LatentAttention`, `MoE`, `MLP`,
`RMSNorm`, `Embedding`, `LMHead` and `DecoderBlock` as ordinary modules. Constructors
register global tensor shapes on meta; forward calls the numerical bindings held
by the components.

A component declares tensor axes, engine state requirements and effective model
configuration. `inventory(model)` derives the tensor and state inventory from the
constructed tree. `place(model, placement)` applies placement once and creates a
private production `ops.table.OpTable`; an internal caller can supply an existing
per-build table with `op_table=`.

Placement calls `Component.apply_placement(placement, site=..., op_table=...)`.
The component derives local activation geometry, resolves operations through the
table, and constructs placement-dependent communication seams. Bindings live in
`OpSlots` as immutable `core.bindings.BoundOp` values. Forward uses their retained
callables directly, without table lookup in the token execution path.

Engine-managed attention and recurrent-state operations remain unresolved until
the adapter calls `attach_core`. Fused ownership is explicit in the attached
binding. Adapters must finish required attachments before checks admit the model.
Optional stages absent from an architecture and required unbound stages have
different meanings; a missing required stage must fail setup.

`resolved_ops(model)` reports retained bindings, and `verify_placed(model)` reports
unresolved slots. Checks derive execution records from those bindings and compare
their numerical identity. Inventory and inspection skip unexecuted subtrees for
execution evidence while retaining their loaded and synchronized parameters.

Reference tests explicitly pass `OpTable(mode="reference")` to placement. Production
selection does not fall back to reference arithmetic.
Production components import no test oracle or runtime implementation.

The dependency direction is `models/runtimes → nn → ops → core/schema`. Operation
families receive tensors and explicit setup inputs; cache pools, process groups
and runtime objects are supplied through their appropriate component interfaces.
See [private operation table](../../docs/op-table-redesign.md) and
[execution validation](../../docs/contract-redesign.md) for selection and comparison details.

Primary sources: [component placement](../nn/component.py),
[inventory and inspection](../nn/inspect.py),
[retained operation slots](../nn/_slots.py), and
[private resolver](../ops/table.py).


<a id="preparation-internal-preparation-lifecycle"></a>

### Internal preparation lifecycle

```python
from isoexec.compiler.capture.capture import capture
from isoexec.compiler.partition import partition
from isoexec.compiler.prepare import prepare_model, materialize_stage

model = source.build_meta(raw_config)
captured = capture(model, example_args, example_kwargs)
prepared = prepare_model(source, model, captured)
stages = partition(captured, native_layer_ranges)
local = materialize_stage(
    stages[placement.pp.rank], prepared, placement,
    device=device, checkpoint=checkpoint, op_table=op_table,
)
```

1. Snapshot whole-model inventory, component configuration, logical layers,
   state declarations, OP declarations, checkpoint mapping and sharding facts.
2. Extract stages from the native logical layer assignment.
3. Clone the selected unplaced stage, preserving aliases. The original meta
   model and other extracted stages remain untouched.
4. Plan canonical tensors using actual component ownership, then apply existing
   component placement once. EP pruning retains original expert identities.
5. Verify each held parameter has a plan; allocate local storage and restore
   local tied aliases. Allocate synced buffers from their declared plans.
6. Attach lazy packing hooks, load checkpoint slices and refresh only packing
   owned by the selected stage. Reload resolves bindings against live storage.

`TensorPlanMetadata` lives in the neutral `isoexec.schema` layer. Parameter-owning
components provide `tensor_plan_metadata()` for their direct parameters and
synchronized buffers; ordinary composed models need no additional declaration.
Expert holders state their global ID/count and ETP axis. The generic collector
checks canonical identities, aliases, shape/dtype and shard axes, without a
component-class or expert-role switch. Owner-specific validation runs before
pruning. Plain holders can use the same protocol for new parameter roles. No
`lm.layers`, `.mixer`, or checkpoint-family namespace is
required. New component kinds still need their own correct tensor axes,
placement and OP contracts; writing three author files does not implement a new
distributed component.

The descriptor continues to cross worker boundaries as JSON. Stage cloning is
worker-local object copying, not pickle transport. No full source module is
registered as a hidden child of a materialized stage.


<a id="preparation-shared-storage-primitives-and-refresh-ownership"></a>

### Shared storage primitives and refresh ownership

`isoexec.storage` provides `CheckpointStore`, `attach_packed_experts`,
`attach_packed_proj` and `refresh_packed_storage`. The model path uses these
supported internal primitives; there is no copied reader.

`refresh_packed_storage(stage)` selects packing by module identity in the stage's
tree. An unrelated model with identical FQNs cannot fail this stage's refresh.
The stage's own packing failure propagates and leaves its execution blocked.
Scoped refresh preserves existing buffer addresses and does not advance the
global `SYNC_EPOCH`, which would otherwise authorize an unrelated model's next
forward to repack. Existing global sync calls retain their process-wide behavior;
the shared mutation epoch still detects storage rebinding.
Buffer safety checks use that owner's original buffer objects, allocations,
addresses and geometry. Allocation counters remain statistics: another owner's
valid allocation during a refresh cannot be mistaken for this owner's buffer
reallocation. Actual changes to this owner's captured buffer are still refused.


<a id="preparation-storage-snapshot-contract-for-phase-5"></a>

### Storage snapshot contract for Phase 5

The Phase 4 preparation layer exposes storage validity separately from the
Phase 5 native admission implemented in [RuntimeStage](#phase-5-native-runtime-adapters):

```python
snapshot = local.storage_snapshot()
# A native adapter must separately lower/validate its program and resources.
local.require_storage_snapshot(snapshot)
```

The worker-local `StorageSnapshot` records the owning stage, local update
generation, Tensor object/address/shape/dtype/device/stride, and current OP
binding identities. It retains identity-bearing objects and allocations while
held, preventing identity/address reuse from making a stale snapshot appear
current. Release old snapshots when their runtime program is retired. Snapshots
are not serialized and are not an assertion that native execution is admitted.

- Every `load()` invalidates old snapshots before the first checkpoint write.
  The generation counts local update attempts, including failures; it is not
  the distributed W0/W1 `weight_version` or a packing epoch.
- Loading or refresh failure leaves receipt and execution invalid. A complete
  successful reload permits issuing a new candidate snapshot.
- Direct Parameter/buffer replacement, storage rebinding, added held buffers or
  changed OP bindings also reject the previous snapshot, even outside `load()`.
- A content-only `copy_` update can reuse the graph after successful loading,
  scoped refresh and the adapter's necessary checks. Snapshot invalidation does
  not require another `torch.export` or PP partition.
- Native `RuntimeStage` shares this owner and checks snapshot validity for each
  Python executable entrypoint, including local layouts, runtime/state resources
  and capture guards. Native callers must quiesce in-flight work. Graph pointer
  retirement and distributed update acknowledgement remain Phase 6; revalidation
  alone does not authorize replay of an old CUDA graph.

Production weight updates must pass through their owning update lifecycle;
arbitrary content writes outside that lifecycle do not establish a new verified
weight version. Python view refresh alone does not validate native graph pointers.


<a id="preparation-metadata-ownership"></a>

### Metadata ownership

| Information | Owner / access |
|---|---|
| Canonical names, global shapes, held dtypes, aliases | Global `Inventory` |
| Tensor slicing / EP ownership at a placement | `PlanFacts` and `LocalSpec` |
| Numerical component settings and boundary output policy | Existing `ConfigSnapshot` and component facts |
| Native layer count before PP assignment | `ModelMetadata.layer_count` |
| Cache/state requirements before engine construction | `ModelMetadata.contract`; existing runtime projections consume it directly |
| Declared component slots before placement | Original `Component.slots` on the captured component tree |
| Actually bound OP IDs, implementations, pins and owned operations | `PreparedStage.bound_ops`, read from current `BoundOp` objects |
| Required slots awaiting native attachment | `PreparedStage.missing_ops` |

Held parameter dtype is not a substitute for compute/accumulation/output
precision. Before placement, preserve component declarations. After placement,
read actual bindings; do not infer a kernel's precision from its weight dtype.
Engine resource-dependent slots remain visibly unbound until adapter attachment.


<a id="preparation-execution-and-storage-limits"></a>

### Execution and storage limits

Local trainer stages with TP=ETP=EP=CP=1 and SP disabled can run with matching
capture inputs and bindings. PP extraction can produce multiple such stages;
tests may compose them locally through `compose_stages()`, but that helper is not
native PP scheduling or cross-rank gradient sync.

For engine placement or any distributed activation layout, storage preparation
is allowed when existing components admit that placement. Calling the raw
stage, body or head then raises **before component execution**: the native
adapter must lower and validate activation layouts, capture guards and runtime
slots. There is no public force-ready flag. This is the preparation-to-runtime
boundary, not a permanent TP/SP/PP restriction. Phase 5 now supplies that
[separate lowered runtime view](#phase-5-native-runtime-adapters); the raw preparation view stays
blocked for distributed execution.

The logical graph cannot be reused blindly: a TP head has local vocabulary
width, SP changes token ownership, and arbitrary authored reductions/views may
need different semantics. Original shape/dtype/device guards stay intact.
Phase 5 must reject an unsupported operation with its location and missing
layout contract, rather than removing guards or assuming every operation is
token-wise. Production example inputs must match the native device and capture
contract; moving parameters does not convert a CPU capture into a CUDA capture.
Standard root `input_ids` still accepts int32/int64 and canonicalizes to int64.
As with Phase 3, a direct stage call consumes canonical positional inputs from
`StagePlan`. Whole-model/root binding owns int32 normalization; Phase 5 adapters
must normalize root arguments before binding a stage.

Registered meta buffers require a value source: declare them in
`synced_buffers()` and the weight map. An undeclared meta buffer fails with its
name and the required fix, rather than receiving uninitialized bytes. Already
initialized local buffers are preserved by the low-level materializer, but the
current `ModelSource.build_meta()` contract still requires registered buffers
to be meta. Runtime scratch/cache initialization belongs to the later adapter.
Buffers created by lazy packing hooks remain owned by those existing hooks.
Shared aliases involving a synchronized buffer require a future canonical buffer
contract and are refused; parameter/buffer aliases are refused as well.

Ties within one stage retain the same Parameter object. Ties across PP stages
share a canonical name and load the same checkpoint slices, but have separate
local storage. Phase 6's installed native binding and synchronized receiver
handle cross-stage gradients and updates; local equality is not proof of that
protocol. See [synchronization](#phase-6-ownership-and-synchronization).

Phase 5 Megatron now refuses training for any canonical parameter consumed by
multiple PP stages, even if frozen. Partition still represents the graph and
projects those consumer facts to local StagePlans. Static eval replicas may use
their initial consistent load, but an owner-level generation anchor prevents
local updates/reloads from being re-admitted through old or fresh runtime views.
Same-stage ties and PP1 reload remain supported. See [ownership admission](#phase-5-native-runtime-adapters)
for the explicit boundary. These guards remain when the actual Phase 6 binding
is absent; the partitioner never fabricates native readiness.

Primary sources: [preparation](prepare.py), [tensor facts](sharding.py),
[checkpoint loading](weights.py), [metadata](metadata.py),
[common shard arithmetic](../runtimes/parallel.py), and
[PyTorch Module.to_empty](https://docs.pytorch.org/docs/2.11/generated/torch.nn.Module.html#torch.nn.Module.to_empty).


<a id="checkpoint-conversion-mapping"></a>

## Hugging Face checkpoint conversions and local shards

Authors import `WeightRenaming`, `WeightConverter`, and built-in `ConversionOps`
from `transformers.core_model_loading`. The only weight entrypoint is
`checkpoint_conversion_mapping(config)`. HF `Split` reads unequal split sizes
from the full target parameter shapes. `isoexec.weights` supplies general
`Deinterleave` and `AddScalar` operations for grouped layouts and scalar offsets.

```python
from transformers.core_model_loading import Split, WeightConverter, WeightRenaming


def checkpoint_conversion_mapping(config):
    return [
        WeightRenaming(r"^model\.", ""),
        WeightConverter(
            source_patterns=r"\.attn\.qkv\.weight$",
            target_patterns=[".attn.q", ".attn.k", ".attn.v"],
            operations=[Split(dim=0)],
        ),
        # Other rules must cover the complete model's registered tensors.
    ]
```

1. `prepare_model` parses HF config and validates the returned HF rule types.
   It retains declarations, not checkpoint tensor values or local conversion plans.
2. At load time, IsoExec uses upstream `rename_source_key`: every matching
   renaming runs in order, followed by the first matching converter. Layer and
   expert indices remain in untouched prefixes. Conversion groups are rebuilt
   from checkpoint names with HF natural ordering, including groups whose
   inputs span multiple safetensors files.
3. The entire checkpoint must cover canonical parameters and synchronized
   buffers exactly once. Registered ties resolve aliases to their canonical
   owners. The model's HF `_keys_to_ignore_on_load_unexpected` attribute explicitly
   allows unused raw checkpoint keys by regex; all other unknown keys fail.
   Missing converter source patterns and encoded source dtypes fail before writes,
   including inputs held only by another stage or expert rank.
4. Rename-only sources use `LocalSpec.slice` directly on the safetensors reader.
   A selected converter receives private full CPU inputs, executes upstream
   `WeightConverter.convert`, and validates all of that group's outputs before
   copying any of them. HF operations receive a temporary meta module with full
   target shapes derived from Inventory; no real full model is allocated. This
   lets HF Split operate before TP slicing, rather than on a TP-sharded input.
   Only one group's values are retained at a time. A custom
   operation may mutate its owned inputs without contaminating the checkpoint.
5. IsoExec applies the existing local ranges to converted CPU tensors, casts on
   CPU, and copies into already allocated destination parameters/buffers. No
   full GPU logical tensors or transient CUDA source copies are allocated by
   this loader. Conversion CPU memory scales with a group, which can exceed
   the local shard size. It does not promise minimum source reads for arbitrary
   HF conversions.

Source dtypes are ordinary BF16/FP16/FP32/FP64; raw quantized encodings require a
separate dequantization contract. Destination storage remains declared by the
components. Checkpoint byte hashing and before/after file checks are retained.
PP/EP filtering, TP ranges, physical packing, weight sync, and native process
groups remain owned by IsoExec and the selected framework adapters.

Implementation: [weights.py](weights.py),
[general operations](../weights/operations.py),
[Qwen3.5](../models/qwen3_5/weight.py),
[Transformers 5.20.2](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/core_model_loading.py).


<a id="phase-5-native-runtime-adapters"></a>

## Phase 5: native runtime adapters


Phase 5 connects generated stage programs to Megatron and vLLM. The adapters
execute the authored dataflow and the original component/OpTable bindings.
They do not construct a native replacement model.
CPU results establish the local contracts described below; native scheduling,
CUDA kernels, collectives and graph replay still require qualification.


<a id="runtime-construction-and-input-ownership"></a>

### Construction and input ownership

An explicit `ModelSource` and raw HF configuration select the model for both
native runtimes:

```python
from isoexec.models import ModelSource
from isoexec.runtimes.megatron.factory import build_model
from isoexec.runtimes.vllm.registration import register_native

source = ModelSource("my_models.demo")
model = build_model(
    source, raw_model_config, checkpoint_path=checkpoint, training=True,
    max_tokens=4096, pg_collection=pg_collection, transformer_config_kwargs={},
)
engine_kwargs = register_native(
    source, raw_model_config, checkpoint=checkpoint,
    engine_kwargs={"max_num_batched_tokens": 4096},
)
```

The trainer's process groups and the engine's own parallel settings stay
independent. SkyRL's source/YAML selection and the built-in Qwen packages are
described in Phase 7.

`build_model()` returns the native trainer model; the serving framework constructs
its own model using the kwargs returned by `register_native()`. Worker settings carry the source's
JSON descriptor and separate raw configuration in `additional_config.isoexec`.
No live source module, graph, Parameter or capture Tensor is transported there.

The shared `text_inputs()` provider describes packed one-dimensional token IDs
and `SequenceContext`. It supplies `CaptureInputs.seq` when the authored forward
omits `seq`; existing explicit-seq forwards keep their original arguments.
IDs accept int32/int64
and normalize to int64; the root IDs and `seq.input_ids` share the same Tensor.
Host `max_q`/`max_kv`, packed offsets, positions, devices, optional fields and
dynamic constraints retain their own contracts. Trainer and engine profiles
differ deliberately in CPU offset availability. Declared MRoPE components use
three position axes for text, without inspecting a model-family name.

Native root ownership travels with `CaptureInputs.root_layouts`. The standard
trainer provider declares token IDs and positions CP-local, positions on their
last dimension, and global device/host cumulative offsets. Guarded repeated
Tensor references inherit the same declaration; conflicting alias locality or
token axes fail capture. This input contract belongs to the runtime adapter,
so model authors retain the HF config and model package interface. Generic CP inputs
without a complete tensor-leaf contract fail with the missing paths and a request
for explicit native `RootTensorLayout` declarations.

CP locality is separate from SP sequence-stream ownership: integer token IDs do
not become TP-scattered activations merely because CP owns local token rows.
Layout analysis starts at the declared roots, including their guarded aliases,
so unsupported token-mixing work before embedding refuses before execution.
Global sequence metadata remains global. Exact functional rules validate the
declared axis, broadcasts and dimensions preceding it; an unsupported axis move
is refused rather than guessed from an equal dimension size.

Extra required roots need explicit `CaptureInputs` from
`isoexec.compiler.text`. The provider
does not invent their shape or dtype. Native adapters currently bind the
standard text roots; another root convention needs an adapter that supplies it.

Capture projects meta parameters/buffers and inputs onto one logical-device
FakeTensor mode on an isolated clone. It allocates no dummy model weights and
does not change the source. Export sees flat Tensor/scalar leaves and the
original root structure is reconstructed internally, retaining the actual input
guard and alias contract. `torch.export` plus functionalization remains the
selected frontend because activation `+=` must preserve later alias/view reads.


<a id="runtime-shared-execution-and-layout-admission"></a>

### Shared execution and layout admission

The implementation separates provenance, semantic analysis and graph rewriting:

```text
Captured graph
  -> dataflow.py: generic provenance before PP cuts
  -> PreparedStage + placement
  -> analysis.py using layout_rules.py: read-only semantic checks
  -> immutable LoweringPlan
  -> lowering.py: mechanical graph rewrite
  -> RuntimeStage: binding, admission and execution lifecycle
```

Components declare a closed `TokenLayoutContract` with a `TokenLayoutEffect`
and token axis. Existing components preserve axis 0; unsupported declarations
or axes fail during analysis, before kernel execution. This is internal metadata,
not another author file. The neutral annotation no longer imports runtime code
from the partitioner.

Functional rules use exact ATen overloads and shared geometry-checked handlers.
Tags select pointwise/reduction candidates; they do not bypass broadcast, token
axis or reduction-dimension validation. Analysis records movement, scalar
recipes, output checks and canonical gradient contributions in frozen records.
The rewrite emits those actions without re-deciding layout or gradient policy.
OpTable still owns numerical bindings and existing Seams own communication.

The rewrite records an immutable logical-value to executable-node name mapping.
PP boundary schemas retain logical names while FX can rename validation or
movement nodes. Trainer wire ownership and engine nonfinal gathers resolve that
mapping; split-head inputs use the actual executable names through the same
required-layout lookup. Missing required layout metadata refuses execution
instead of silently treating a sharded value as replicated. Mode profiles keep
their own mapping and do not change the native PP payload names.

`RuntimeStage` owns a lowered execution view over `PreparedStage`; component
objects and their live parameters are shared, and the complete original source
is not registered as a hidden child. Its stage/body/head entrypoints preserve
the original dataflow and verify storage admission before component execution.

Logical shape metadata is retained separately from physical TP/SP layouts.
Parameterless movement components bind the existing scatter/gather seams, so
their actual OP identities remain visible. Decoder components consume the
supported placed layout; the embedding-to-SP transition is inserted once.
Engine PP wires reconstruct full token rows and scatter again locally when
needed. Trainer PP wires retain the admitted local stream layout.

Ordinary token-wise operations, aligned broadcasts, non-token reductions and
token-preserving views can retain SP. Token mixing, unsupported view geometry,
unknown component layout contracts and functional use of an uncommunicated
parameter shard fail with the operation/module location. Replicated parameter
gradient ownership is checked across every use of the canonical Parameter,
including component and functional contributions. A native finalizer cannot
apply one TP reduction to both partial and already-complete contributions;
that mixture is refused with the exact uses. PRESERVE components propagate
provenance until consumers determine ownership: a broadcast norm can receive
partial gradients, while a scattered full-token norm uses the existing complete
adjoint. Lifted parameters/buffers/constants retain export Tensor metadata so
layout analysis sees their operands and inserts needed scatters. CP likewise uses existing
component semantics; unsupported authored operations get a specific diagnostic.
No singleton-only axis policy replaces component/native admission.

Gradient liveness is derived from the complete author graph before PP cuts.
Tuple/list selections use exact ATen output-adjoint rules: integer indices and
native normalization saved statistics have no adjoint, and detach/factory
operations stop provenance. An unknown selected floating auxiliary output is
reported at native trainer TP admission; it does not make PP partitioning or
ordinary TP1 execution invalid. Only resolved partial contributions enable
`reduce_grad_over_tp`; complete-only uses do not inherit a detached component's
SUM declaration. SP classification still follows physical token ownership.
The Megatron host binds its training profile's policy before native finalizer
construction. Evaluation does not rewrite that cached Parameter set, and adding
a different training policy later refuses before changing model mode.

Pure scalar shape cuts use root metadata recipes and are recomputed locally.
They are not transported as detached activation tensors. Data-dependent or
unbacked scalars without such a recipe are refused. Global root dimensions
remain global when used as numerical data; physical local dimensions are used
only for the admitted layout transformation.


<a id="runtime-megatron-producer-and-consumer"></a>

### Megatron producer and consumer

The Megatron factory constructs a real `TransformerConfig` carrier from global model
metadata, uses native layer-count/offset functions, then partitions the captured
graph. The carrier does not create native TransformerLayers. Current assignment
support is a contiguous native PP cover; VPP/custom chunks need an explicit
native chunk-assignment contract.

`V2MegatronModel` implements native `set_input_tensor` and packed THD forwarding.
Non-final stages pack all live boundary tensors into one differentiable
`[stream_tokens, 1, feature_width]` envelope. Values must have the same token
extent and carrier dtype, with fixed trailing dimensions and a compatible wire
layout. Native variable-length shape exchange carries the true feature width;
it is not assumed equal to `config.hidden_size`. The next stage splits the
features back into the original named values, preserving autograd edges.
Incompatible cuts raise `PP_BOUNDARY_UNSUPPORTED` with the values and remedy.

The terminal authored LMHead returns local vocabulary logits. Padded TP shards
keep equal physical width and mask columns beyond the logical vocabulary before
the existing trainer logprob helpers consume them. Whole-stage native reentrant
checkpointing preserves the stage parameters' gradients, including a first
stage whose model inputs are integer IDs.

Native DDP/optimizer construction reuses the existing wrapper. SkyRL packs
input token IDs on the first pipeline stage and for vision-language models.
Later IsoExec pipeline stages consume the received hidden states and the batch's
packed sequence metadata; their `input_ids` may be `None`. The integration binds
scoring to the model's registered LMHead and preserves the native deferred
gradient-finalization order.


<a id="runtime-cross-stage-parameter-ownership-admission"></a>

#### Cross-stage parameter ownership admission

The partitioner records each local stage's `cross_stage_parameters` from the
completed global parameter consumer map. These immutable canonical-name/stage
pairs describe consumers, not gradient or emission ownership. Buffers are
excluded; ordinary TP replication does not become a cross-PP consumer fact.
Partition still represents such graphs and does not apply training policy.

Phase 6 carries topology-independent CanonicalTensorOwnership plus partition-local
PlacementEmissionPlan facts, and installs canonical ParameterSync over NativeOptimizerBinding
before releasing cross-stage training admission. Pending construction is not
readiness: native DDP, optimizer groups, replica values and gradient storage must
be validated. Without that binding, the generic Phase 5 refusal remains, including
frozen parameters. See [synchronization](#phase-6-ownership-and-synchronization).

Cross-stage replicas may run static eval from their initial consistent checkpoint
load. First successful eval admission anchors the prepared owner's generation.
Without synchronization, runtime storage updates refuse before writes; direct owner reloads revoke the
snapshot and cannot be locally re-admitted, even through a fresh runtime view.
Same-generation revalidation is allowed. PP1 reload is unaffected. Initial load
consistency still relies on the same checkpoint/source contract; these local
guards are not a distributed acknowledgement or mutable-version protocol.
Arbitrary raw Parameter writes outside the owning lifecycle are unsupported.

Local graph/RuntimeStage composition can serve as a mathematical gradient oracle.
Manually summing canonical contributions does not demonstrate native tied-gradient
collectives and cannot bypass the production Megatron gate.


<a id="runtime-canonical-emission-names"></a>

#### Canonical emission names

Name emission unwraps framework wrappers to the logical stage module, then uses
actual held TensorBindings and global Inventory tie declarations: only the declared tie owner emits its
canonical name, aliases do not, and buffers are excluded. Generic cross-stage
parameters use the explicit first-consumer policy only after synchronization
installation; otherwise they refuse. Physical holders are not automatically emission owners.

Stages gain no synthetic `.cfg`.
This protocol enumerates names only. CanonicalTransfer and the synchronized
receiver separately provide bounded byte transport and version acknowledgement.


<a id="runtime-vllm-producer-and-consumer"></a>

### vLLM producer and consumer

vLLM registration installs independent dense/hybrid wrapper classes and the
existing custom loader. Pre-instance GDN shape/dtype queries use the same global
ModelContract. Resource binding follows original StateRef FQNs and live storage
callbacks; attention/recurrent cache modules own only their native resources.
Construction failure and close remove only that owner's registrations.

The plugin is the sole class-installation owner for API registration and fresh
workers. Repeated successful same-process installation is idempotent. Both the
plugin and worker bootstrap publish installed flags only after complete success,
so a failed attempt propagates and a later invocation retries rather than
returning kwargs backed by incomplete registrations.

Each non-final stage returns `IntermediateTensors` containing named live tensor
values and canonical root token IDs. vLLM allocation/selection requires each
field to retain the scheduled token dimension and fixed trailing dimensions.
The final body returns one terminal-head Tensor for the runner's row selection;
`compute_logits()` calls the original head once on those rows. Vocabulary shards
are gathered before the original logical-vocabulary trim. Sampled rows are
already replicated and are not gathered a second time across the token axis.

TP, PP, engine DP, SP and supported component EP settings retain their values.
The current cache providers require BF16, CUSTOM attention, raw logprobs and the
existing runner/backend envelope; incompatible native options fail explicitly.
Text-only execution requires token IDs, not native prompt embeddings. Admission
of a configuration is not proof that its distributed native execution succeeds.


<a id="runtime-pre-instance-facts-cache"></a>

#### Pre-instance facts cache

Registration classification and native state shape/dtype queries share a
process-local cache of immutable `PreInstanceFacts`: portable state/OP contract,
layer count, declaration digest and capture profile. It retains neither
`ModelMetadata`'s mutable containers nor model/Tensor/GraphModule/native resource
objects. Cache misses still perform complete capture and validation; worker-local
CUDA execution capture is independent and unchanged.

Keys preserve the raw configuration's mapping order and Python types, plus the
source descriptor, input profile/bounds and ambient capture modes/defaults.
Native ranks, process groups, TP, convolution layout and min-pages policy are not
cache keys; cache-specific projections apply those after reading global facts.
The bound is eight entries. One miss lock prevents concurrent duplicate export,
failed attempts publish nothing, and fork resets both facts and lock.

Author code/dependencies and construction-affecting process setup remain fixed
during a process lifetime; changing them requires clearing the private cache
at a quiescent boundary or restarting. Package name
and declaration digest are not source-code fingerprints. This cache does not
establish cross-worker identity/version agreement. The pinned default-device
inspection reads metadata only and bypasses reuse if it cannot resolve the
context; lookup does not initialize CUDA.


<a id="runtime-storage-and-failure-lifecycle"></a>

### Storage and failure lifecycle

The construction factory performs final admission once, after the adapter has
completed native setup and validated its calling convention. Later DDP and
optimizer storage transitions still require fresh admission; they are actual
updates rather than repeated initial construction.

Admission requires complete loading, matching local shape/dtype/layout and
aliases, unchanged captured component facts/mode/trainability, executable OP slots, and native
resource attachment. Metadata-only abstract bindings cannot execute. Packing is
prepared before taking the snapshot, so first execution does not change held
buffer identities unexpectedly.

Changing a Parameter's `requires_grad` requires rebuilding the captured gradient
contract; revalidation cannot silently turn a frozen path into an unexamined
partial-gradient path. Direct root Tensor references in executable programs must
also match the authoritative prepared owner, as must authored module references.
Replacing such a Parameter/buffer or a separately held root module
requires rebuilding the runtime view; a fresh owner snapshot cannot endorse an
old program's copied root reference. The same check rejects replacement through
the execution view rather than the prepared owner.

`RuntimeStage.storage_update()` revokes the shared owner's admission before a
quiescent storage transition, refreshes only that owner's packing, validates the
result and issues a fresh snapshot on success. DDP bucket ownership retires
stale packing aliases without rebinding foreign parameter storage. Failure
leaves all old views blocked until complete reload; a sibling view is never
silently re-admitted. Native callers must quiesce in-flight execution. Optimizer
completion runs after the framework's step and requires quiescence throughout
the step; it does not retroactively fence earlier writes. Phase 6's optimizer
entrypoint instead wraps the actual native step and completion in one transaction.

Megatron releases packing through
`isoexec.storage.release_packing_to_storage_owner()`. The storage facade uses
existing projection/MoE provider APIs and architecture dispatch. Scoped prepare
and refresh also resolve through the existing facade. The same bookkeeping
module, mutation/sync epochs, allocator and hooks remain shared; no state is
copied and no new accelerator support follows from this API cleanup.

`revalidate()` establishes Python-entrypoint admission only. It does not retire
or recapture a native CUDA graph that holds old pointers, and replay bypasses
Python guards. Storage rebinding therefore cannot authorize reuse of such a
graph. Native graph retirement/reapply and distributed version acknowledgement
use the Phase 6 synchronization owner. Stable-copy native updates preserve
addresses; pointer/layout changes fail before writes and require a new native
graph lifecycle. Content-only scoped updates do not themselves attest W0/W1.


<a id="runtime-remaining-integration-and-proof"></a>

### Remaining integration and proof

Phase 6 implements the ownership, transport and transaction contracts described
in [synchronization](#phase-6-ownership-and-synchronization). Direct source-less CPU fixtures retain
their guards. Source/CPU evidence does not qualify native gradient, transport,
allocator or CUDA replay behavior. Cross-PP checkpoint equality is not a gradient
synchronization proof.

Qwen3.5-0.8B PP1 and PP2 native Modal runs completed, each with W0
comparison, one optimizer update and W1 comparison. See the
[qualification snapshot](#skyrl-native-qualification-snapshot)
for the exact source/runtime/hardware scope and remaining W1 bitwise differences.
PP4 and larger-model native qualification are deferred. CPU adapter analogues
alone do not qualify native PP, DDP, collectives or CUDA graph replay.

Primary implementation sources: [source](source.py),
[text inputs](text.py), [capture](capture/capture.py),
[partition](partition.py), [realization](realization/),
[metadata recipes](capture/values.py), [Megatron adapter](../runtimes/megatron/factory.py),
[vLLM adapter](../runtimes/vllm/factory.py),
[vLLM registration](../runtimes/vllm/registration.py), and
[PyTorch export functionalization](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/export.html).


<a id="phase-6-ownership-and-synchronization"></a>

## Phase 6: ownership and synchronization


Phase 6 connects the authored graph to existing Megatron gradient/optimizer
owners and SkyRL IPC/NCCL transports. Authors provide only `config.json`,
`model.py`, and `weight.py`; no rank-specific tensor names, sharder or stage
definitions are required.


<a id="synchronization-canonical-ownership"></a>

### Canonical ownership

Partitioning derives immutable `CanonicalTensorOwnership` records from the
unpartitioned Inventory. The authored canonical name or declared tie owner is
the logical owner; changing PP never changes that identity. For every consumed
tensor, `PlacementEmissionPlan.tensor` references the corresponding record, and
every StagePlan carries the same emission plans assigning consumer stages,
local aliases and one physical emitter for this partition. A held declared tie
owner wins; named explicit owners must identify exactly one held consumer,
otherwise partitioning fails. Implicit sharing may use `first_consumer` for the
emitter only. Nonconsumer stages retain the emission plans so native collectives
follow the same schedule. Inactive checkpoint-only tensors retain logical
identity in the global Inventory without a fabricated physical emitter.

Name emission and byte transfer have separate structural host protocols.
`CanonicalTransfer` preserves the host through framework wrappers, reuses
existing LocalSpec slicing/assembly and PP gathering, and yields bounded
streams. Every receiver holds all of its declared aliases even when it emits
none. Parameters, frozen parameters, and declared synchronized buffers are
included. EP recutting uses declared expert metadata rather than module-name
patterns. `TransferPlan` combines emission ownership and source/destination
LocalSpec to record reuse, slicing, TP/ETP gather or EP exchange before execution.
The native wire planner declares transport capabilities and the fallback reason.
Every full-stream fallback, including dense transfer, requires a qualified
uniform collective schedule. External full providers replace TP/ETP reconstruction
and do not depend on unrelated initialized native groups; EP exchange retains
its separate existing native qualification. Plans contain only executed actions,
not fabricated or unused physical routing metadata.
The current SkyRL EP transport has no qualified expert-shard routing schedule;
it therefore uses the uniform full logical tensor stream and existing collectives.
This is a qualified correctness fallback, not efficient frontier-MoE resharding.
Exact cross-EP producer/receiver routing requires a separately qualified native
transport implementation; an alternative capability flag alone is not support.


<a id="synchronization-native-trainer-installation"></a>

### Native trainer installation

`build_source()` prepares a pending ParameterSync when cross-stage parameters
exist. The host cannot execute until native DDP and optimizer installation
complete. Replica flags are installed before optimizer construction. Actual
optimizer groups and initialized replica values must agree; alias-dependent
optimizer settings are not silently normalized.

Canonical `ParameterSync` depends only on `NativeOptimizerBinding`: gradient and
master images, statistics ownership, optimizer options, native finalization,
copy-back attestation and installed-resource lifetime. The Megatron adapter owns
its private master maps, bucket fields, `main_grad`/`main_param`, replica flags
and forced all-reduce details. The generic algorithm does not inspect native
optimizer representation. A second binding must satisfy the same complete-gradient
and master-image contracts; implementing the protocol does not qualify its backend.

The existing native finalizer runs once. Shared replicas require full native DP
reduction, including overlapping backward hooks, before existing TP/SP reduction
and the canonical PP SUM. Nonconsumer ranks contribute zeros in the same global
schedule. `main_grad` storage is preserved. Native embedding-only PP reduction
is disabled for this generic path to avoid reducing a tied parameter twice.

`shared` replica flags keep duplicate parameters out of native gradient norms.
The actual native optimizer updates every replica. Afterwards both model values
and FP32 master parameters must agree; broadcasting BF16 parameters cannot hide
a divergent optimizer master. Multiple distributed optimizer instances are
refused for cross-PP sharing because their native inter-instance reduction does
not supply the required complete gradients.

After native copy/gather completion, `parameter_sync.after_step()` verifies the
model values, replicas and FP32 masters that belong to its cross-stage ownership
records. The optimizer path otherwise trusts native optimizer copyback and does
not run the global `copyback_errors()` check after every update. IsoExec therefore
does not guarantee per-step detection when an ordinary parameter's authoritative
FP32 master changes but its live model storage stays stale. A configuration with
no installed `ParameterSync` has no post-step master-to-model attestation.
Unsupported precision-aware, quantized or stochastic-copy policies still refuse
before writes. Native copyback and parameter gather completion remain the
optimizer's responsibility; the retained cross-stage checks never repair stale
model weights silently.

Storage transition preparation and readiness publication are separate. While
every owner remains `UPDATING`, once-only local preparation finishes packing,
admission, storage snapshots and binding reads. Native optimizer, residency and
already-synchronized DDP wrapping callers prepare all local owners, then exchange
errors through the actual execution-owner groups before any owner publishes
`READY`. Context exits publish only previously prepared facts. A failed local
preparation is retained rather than retried during cleanup; every affected owner
stays blocked. Initial model construction retains its separate startup admission.

Copy-policy and copy-back failures use explicitly supplied native DDP `tp_group`,
`pp_group` and `dp_cp_group` lifecycle owners. Gradient-statistics groups retain
their mathematical purpose: a non-distributed optimizer's model-parallel norm
group does not prove DP lifecycle coverage. Missing execution-owner groups refuse
actionably; there is no fallback to singleton statistics groups. Recoverable
agreement assumes intact admitted groups and quiescent owners, and does not claim
recovery from process loss or a broken CUDA/communication context.

Residency explicitly supports existing bare native reference/inference hosts
with no optimizer. Those hosts own `pg_collection.tp`, `pp` and `dp_cp`, while
training DDP owns its separate `tp_group`, `pp_group` and `dp_cp_group` fields.
The native adapter resolves the selected owner role and checks all required
groups and local membership before storage transfer begins. It does not create
a fake DDP wrapper, infer global groups, or reuse norm-statistics communicators.
Optimizer-update callers still require their actual DDP owner.

Canonical validation may itself communicate only within one PP subgroup.
Before-step refusals are therefore collected and exchanged across the actual
execution mesh before any rank can enter an optimizer write. The same mesh
agreement follows canonical post-step and residency-refresh validation before
local commit preparation/publication. A PP-local refusal cannot leave unrelated
TP peers ready to write or waiting at the following execution collective.

The SkyRL optimizer entrypoint fences **before** `strategy.optimizer_step()`.
It drains native copy/gather completion and verifies replicas before source
execution/export becomes ready. A post-step callback alone is insufficient.
Trusted trainer offload/onload uses a separate residency transaction. The native
owner is responsible for preserving model and optimizer values; IsoExec does not hash
their contents before and after every transfer. Residency retains execution
blocking, pending gradient preservation, binding/admission validation and
distributed failure agreement. An offloaded owner cannot execute or export.
Rebinding native gradient storage requires the installed owner authority;
arbitrary replacement followed by `admit()` remains invalid.

Trainer weight reload after optimizer construction is explicitly refused until
an optimizer-master-aware reload contract exists. Eager PP1 supports complete,
strict canonical reloads. This restriction does not disable native optimizer
steps or any supported placement axis.


<a id="synchronization-definition-and-checkpoint-identity"></a>

### Definition and checkpoint identity

`DefinitionIdentity` factors installed author-package bytes, normalized semantic
graph, component implementation, and module-level dependency artifacts. The package
digest includes helpers beyond the three role files, with stable relative names
and cache exclusion. Semantic identity includes normalized authored operations
and constants, logical layer mapping, state declarations, semantic input
dtype/rank contracts and global declarations. Component forward source remains
separate from the semantic graph.
Lifted immutable Tensor constants include their actual bytes. Source package or
ConfigSnapshot alone is insufficient: changing an arithmetic coefficient or
constant must change identity even when OP bindings remain identical.

Capture guards, native movement, and device indices are excluded from this
shared definition identity. Trainer/engine profiles retain their own exact
export guards; native CPU offset convenience does not change logical sequence
semantics. Standard token IDs accept int32/int64 and normalize to int64. Other
Tensor inputs retain exact dtype/rank identity.

Initial safetensors artifacts receive a streaming content fingerprint with a
1 MiB read buffer and file-change checks. Absolute checkpoint paths are excluded;
artifact names/content are retained. This is an artifact identity, not a proof
that differently packaged checkpoints are logically equivalent. CheckState and
the PP census require the same initial fingerprint. Subsequent versions use
canonical emitted/applied byte digests instead of claiming that the initial
checkpoint still describes updated weights.

Existing numerical recipe/twin-class checks, toolchain checks and OP engagement
remain independent. Precision records use held weight dtypes, component boundary
output dtypes, actual activation policy, and the qualified MM recipe identity.
Placement-scoped movement cannot invent authored-call coverage. The full PP
census covers every authored component occurrence once, including repeated
stateless calls. Dependency provenance covers unconditional module-level imports and actual
module/imported callable bindings. Conditional primitive imports and arbitrary
dynamic or transitive Python import closure are not claimed. Ambient loaded
modules never turn unexecuted type-only imports into dependencies. Immutable wheel RECORD
manifests identify installed distribution artifacts; local/editable dependencies
use package bytes. RECORD provenance assumes installed wheels are immutable and
does not attest arbitrary edits to their live files. Native dependency/toolchain
and recipe checks remain independent. The compatibility wire key `source_identity`
contains the composite DefinitionIdentity digest; peers also compare the factor
manifest. This is scoped definition provenance, not a claim to fingerprint an
entire running process or prove numerical equivalence.


<a id="synchronization-inference-update-transaction"></a>

### Inference update transaction

```
READY -> UPDATING -> APPLIED -> VERIFIED -> READY
            |           |          |
            +-----------+----------+----> FAILED (execution blocked)
```

Application is not readiness. The receiver validates the source/version,
canonical stream coverage, all held aliases, and source/applied byte digests.
It copies into existing storage and refreshes only the current owner's packing.
Native object/address/shape/dtype/stride assumptions must remain stable; an
unsafe replacement fails before writes and requires a new native graph/storage
lifecycle. Python revalidation does not retire or rebuild a CUDA graph.

Weight-byte auditing uses `isoexec/weights/2`, replacing the former integer
checksum. Each 32-byte leaf is SHA-256 over `isoexec/tensor-bytes/2\0`, an
eight-byte big-endian metadata length, UTF-8 JSON metadata, then actual tensor
bytes in logical C order. Metadata has sorted keys and compact separators;
`name`, `dtype`, `shape`, `nbytes`, `order="C"`, and host `byteorder` are included.
The root hashes `isoexec/weights/2\0`, an eight-byte big-endian leaf count, and
the leaves sorted by canonical name. Tensor arrival order, storage strides,
offsets and chunk boundaries do not change the digest. Empty tensors retain
their metadata identity; NaN payloads and signed-zero bits remain distinct.

The wire root remains 32 bytes (64 hexadecimal characters); public tensor
leaves are now 32 bytes rather than 16. Mixed-schema peers refuse a digest
mismatch with no legacy-checksum fallback; all participating processes must
be rebuilt together. Cross-endian peers also disagree explicitly. Local
copy attestation directly compares byte views while source data is available,
and retains a pre-copy cryptographic witness even if the incoming view aliases
destination storage. Finish audits use the retained SHA-256 leaves.

Post-wake verification does not rehash weights. A stale sleep backup
overwriting newly synchronized bytes is a sleep/wake ordering hazard, and the
framework owns that order: `wake(weights) -> weight sync -> wake(kv_cache)`.
After wake the receiver checks held tensor/alias identity and the native pointer
layout, then returns the receipt whose digest was verified at apply. This
intentionally drops a byte-level read-back; regression and qualification tests
cover the transfer and runtime behavior instead.

Hashing reads every audited device byte to the CPU in at most 4 MiB pieces,
including non-contiguous column slices; it never flattens/copies a full large
tensor first. This bounds staging memory but introduces CPU hashing, D2H
traffic and synchronization costs. It is a correctness implementation, not a
claim of the old GPU checksum's throughput. CUDA cost still requires measurement.

SkyRL owns server fan-out, pause/resume, weight transfer and waiting for RPC
completion. Its adapter parses the framework `body/results` envelope and pins a
native `ParticipantManifest` for sleep/wake and final response coverage:
independent expected RPC slots and immutable `(server, process ID, GPU UUID,
slot, named coordinates)` members. This validates replies from the
framework-owned operation; it is not an IsoExec permit or a second transaction.
The vLLM 0.28 adapter projects actual executor slots and initialized WORLD/group
membership without deriving EP rank, TP-times-PP worker count or
one-DP-rank-per-server packing. Actual trainer completion receipts supply the
source run/version and emitted digest; a cache-salt counter is not substituted
for that version.

The production order is:

1. Pause generation; gate every execution owner UPDATING before memory changes.
   Use the native Executor to sleep before trainer onload, then wake weights.
2. Run the existing sender. Each receiver checks its complete canonical stream,
   copies in place, and remains locally APPLIED.
3. Restore KV memory without resuming scheduling.
4. Await prefix-cache reset on every server; check held tensor/alias identity
   without rehashing weights. The same worker RPC runs `runtime.revalidate()`,
   records the applied digest/version, and changes VERIFIED to READY locally.
5. Each EngineCore requires every native worker reply to identify one READY
   source version before setting and reading back its native weight version.
6. SkyRL waits for every server reply. The adapter checks response coverage and
   producer digests, then SkyRL resumes generation and advances the cache salt.

No worker waits for another worker's permit. Independent engines can share
coordinates; the final response check still uses server, process ID, slot, GPU
UUID and numerical coordinates so a missing or foreign reply cannot be counted.
Failure aborts reachable local receivers and leaves generation paused; a failed
receiver remains FAILED and cannot execute even if framework cleanup later
attempts to resume.

The residency policy consumes `NativeResidencyOwner`; all EngineCore patching and
private Executor fields live in the version-specific vLLM 0.28.0 adapter. Reserved
RPCs refuse other versions. Sleep/wake is routed through native Executor. This preserves worker
buffer/GDN hooks and executor `sleeping_tags`/`is_sleeping` bookkeeping. Native
vLLM `EngineCore.wake_up()`
may resume scheduling when the last memory pool returns, which would precede
local completion unless the framework retains its pause. Failure aborts all reachable receivers and leaves generation
paused; it never rolls partially written weights back or resumes in `finally`.
Recovery requires a complete new stream. A failed weights-only wake first
completes the remaining native wake while gated, then sleeps afresh; native
Executor.sleep would otherwise skip a partially sleeping owner. IsoExec admits the existing IPC/NCCL path;
native RDT/delta reload requires another receiver contract and is refused.

The residency owner declares its managed resource set. Generic policy receives
an opaque, already-validated sleep request or declared wake resource names;
it contains no vLLM pool names, `tags`/`level` wire fields or numeric sleep policy.
The vLLM adapter validates its original level/tag RPC before any memory change.
Partial recovery and complete sleep/wake remain checked while scheduling is paused.

The readiness check is installed before native GPUModelRunner execution,
including CUDA graph replay; checking only Python stage forward is insufficient.
Stable in-place updates preserve graph addresses. Arbitrary pointer/layout
replacement is refused rather than clearing unrelated graphs globally.


<a id="synchronization-evidence-boundary"></a>

### Evidence boundary

CPU checks cover source/constant negative controls, real canonical storage and
receiver RPC lifecycle, physical response coverage/failure ordering, and Gloo
collectives with a native-free DDP/optimizer fixture. They do not prove native
Megatron scheduling, NCCL, CUDA kernels, allocator sleep/wake, or graph replay.
Historical topology receipts do not qualify a new dataflow.

The Qwen3.5-0.8B text-only/MTP-off package and SkyRL configuration completed
native PP1 and PP2 runs on Modal, each with W0 comparison, one real
optimizer update and W1 comparison; no W2 was produced. See the
[qualification snapshot](#skyrl-native-qualification-snapshot)
for independent artifact verification and remaining W1 bitwise differences.
PP4 and larger-model native qualification are deferred.

Primary sources: [weight transfer](../sync/weights.py),
[PP gather](../sync/pp_gather.py), [Megatron model](../runtimes/megatron/model.py),
[vLLM model](../runtimes/vllm/model.py), [weight source](../sync/weight_source.py),
and [vLLM 0.28.0 EngineCore](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/engine/core.py).


<a id="phase-7-skyrl-and-model-package-integration"></a>

## Phase 7: SkyRL and model-package integration


<a id="skyrl-native-qualification-snapshot"></a>

### Native qualification snapshot

On 2026-10-04, Qwen/Qwen3.5-0.8B at revision
`2fc06364715b967f1860aea9cf38778875588b17` completed four real SkyRL runs,
comparing this model path against the previous implementation at PP1 (4 H100s)
and at PP2 (8 H100s). Each used
Megatron TP2, dense DP2 and SP; two independent vLLM engines each used TP2,
the matching PP degree, native DP1 and no SP. Trainer/inference were colocated.
CP, EP, ETP, PCP and DCP were 1. Execution was text-only, BF16, MTP/speculative
decoding off, LoRA rank 0 and unquantized. Each run completed W0 comparison,
exactly one native optimizer update, W1 sync/readback/ACK and W1 comparison.
Negative controls, nonzero outputs, complete vocabulary coverage, native
lifecycle evidence and independent downloaded artifact verification passed.

For both topologies, the two implementations' W0 weights and all 362 finalized gradient tensors
were bitwise equal. W0/W1 full logits and FP32 full logprobs were exact across
128 valid response rows per topology, each covering all 248320 vocabulary
entries; trajectories and training batches also matched. W1 weights retained
9 BF16 element differences at PP1 and 6 at PP2, each one BF16 ULP. The original
full-weight bitwise gate remains false. Native grad norms differed by one FP32
ULP, placing the remaining investigation at native norm/clipping/update; the
specific reduction-order cause was not independently replayed. No kernel or
topology change was made to conceal these differences.

The qualified immutable source bundle is
`89f34aedebe8fa04d97d2d468f8e5922595077dcdd5963e7dd64e5fa5c95958f`.
Complete source, resolved configs, weights, gradients, logits, full logprobs and
receipts were retained outside the repository. GPU claims apply only to the
recorded source/runtime/H100 topologies. PP4, larger models and MTP-on remain
unverified; CPU partition coverage does not expand this native qualification.


<a id="skyrl-source-configuration"></a>

### Source configuration

`ModelSource` identifies an importable author package. New packages use HF
configuration classes and select the `model.py` class through `architectures`.
Packages retain `weight.py` and load HF or local
safetensors weights. A source descriptor contains no executable object or model
instance and does not deploy the package; install it in every worker environment.

Select the model package in SkyRL:

```yaml
isoexec:
  model_package: isoexec.models.qwen3_5
  revision: null  # Optional Hub commit, tag or branch; resolved once in the driver.
  raw_config: {}  # Recursively merged HF JSON overrides.
  trainer_placement:
    leaves: 8
    leaf_dtype: bf16
  inference_placement:
    leaves: 8
    leaf_dtype: bf16
trainer:
  ckpt_interval: -1
  hf_save_interval: -1
  resume_mode: none
  policy:
    model:
      path: Qwen/Qwen3.5-0.8B
```

The dense source is `isoexec.models.qwen3_5`. Both sizes use
`Qwen3_5ForConditionalGeneration`, with separate configurations in
their selected Hugging Face checkpoint configs. The 9B model has
32 decoder layers and an untied head; 0.8B has 24 layers and a tied head.
Both execute the text decoder with vision and MTP excluded.

`isoexec.models.qwen2_moe` defines `Qwen2MoeForCausalLM`, matching
its HF architecture and vLLM module names. Qwen1.5-MoE keeps its configuration
in the selected Qwen1.5-MoE checkpoint config. It retains per-layer router outputs for the existing
auxiliary-loss integration. All built-in models use HF configuration objects,
the HF checkpoint conversion contract, and implicit sequence context.
Native GPU qualification remains separate from CPU construction/loading checks.

For a Hub ID, the driver resolves the requested revision to one commit SHA and
downloads config, safetensors, tokenizer assets and Python code into that
immutable snapshot. It rewrites the policy and enabled reference paths before
the experiment constructs its tokenizer or workers. Repeated resolution uses
the already selected local path. Workers do not independently resolve a moving
Hub branch. Parsing belongs to the selected package;
source selection does not route by checkpoint model family. The weight loader
reads the same local artifact. Local safetensors files retain their exact weight
file identity and use config/tokenizer files from their containing directory;
the native engine receives that directory for configuration and the exact file
as its checkpoint source.

Mount the snapshot cache at the same absolute path on every node, or stage the
local artifact there before worker startup. An unavailable path is an actionable
error; it never falls back to downloading moving `main` in a worker. Fully
qualified external `auto_map` code references require a separately pinned local
artifact and are refused by this Hub snapshot adapter.


<a id="skyrl-model-owned-configuration-overrides"></a>

### HF configuration overrides

`isoexec.raw_config` is recursively merged into the checkpoint's
HF configuration. Nested mappings retain unspecified fields; lists, scalars and
other JSON values replace the corresponding base value. For example,
`{"text_config": {"mtp_num_hidden_layers": 0}}` changes that field while retaining
the other `text_config` fields. No Python configuration hook is imported.

Both inputs are copied and must contain ordinary finite JSON values with string
keys. Custom objects, non-string keys, NaN and infinity are refused.

The driver resolves each configuration directory once per source/patch/checkpoint
request and base configuration contents. It rereads the base JSON and compares its
ordered digest before reusing the resolved record. Editing a local `config.json`
at the same path therefore creates a new request. Policy/reference checkpoints
sharing a directory share that resolution. The existing `isoexec` JSON
mapping receives the record only after all artifacts and configurations validate.
Do not write the internal `_resolved_source_config` field yourself.

Workers consume private copies of the resolved JSON without merging again or
reading replacement configs. Megatron construction, vLLM's `raw_model_config`,
its pre-instance cache and `DefinitionIdentity` consume the same values. A changed
source, patch or checkpoint path makes an old record stale; repeat driver
preparation before starting workers for the new request.


<a id="skyrl-trainer-capture-and-persistence"></a>

### Trainer capture and persistence

Canonical weight synchronization supports rollout updates. Native training
checkpoint save/restore and HF-format export remain unsupported: arbitrary
authored weight transformations do not supply an inverse/export contract.
SkyRL rejects positive `trainer.ckpt_interval` / `trainer.hf_save_interval`,
checkpoint resume requests and `resume_path` during driver resolution, before
workers start. Disable them as shown above.

For standard batched rollouts, the adapter derives the trainer capture token
bound from `trainer.max_prompt_length`, `generator.sampling_params.max_generate_length`,
both train and prescore microbatch sizes, the native soft token-packing budget
and TP/CP alignment. It passes the resolved bound to capture instead of using
the local API's default of 4096. An over-budget single sequence is not truncated.

Custom/unbatched trajectories need an explicit positive packed bound:
`isoexec.trainer_capture_max_tokens: 12000`. This bounds *padded tokens in one
train/prescore microbatch*, rather than a single request or the whole rollout.
It must be at least 4 for capture. The resolved value is deployment metadata in
`IsoExecConfig.trainer_capture_max_tokens`; it does not change the three author
files. Runtime input guards still reject larger actual tensors and ask the caller
to rebuild the capture with a sufficient bound.

Megatron's default PP assignment requires the decoder-layer count to be divisible
by PP. The adapter reports that policy before native assignment; unequal first/
last stages and custom native layouts are not exposed by this adapter.

Both placement policies are optional. `isoexec` accepts the
`trainer_placement` and `inference_placement` keys, but the native runtime models
do not read them. Native parallelism is configured through SkyRL's Megatron and
inference-engine fields, including their TP, PP, DP, EP and SP settings.
Unsupported dataflow/layout/native combinations return an admission error rather
than silently reducing a degree.

Trainer construction uses `isoexec.runtimes.megatron.factory.build_model()` and vLLM
construction uses `isoexec.runtimes.vllm.registration.register_native()`. SkyRL passes only the source descriptor and raw configuration to engine workers.
Providing `isoexec` selects native IsoExec runtime models; an empty mapping or missing
`model_package` is refused before workers start. Omit the mapping or set it to `null`
to keep native SkyRL. The integration targets Megatron training
and vLLM inference.


<a id="skyrl-training-and-prescore-evaluation"></a>

### Training and prescore evaluation

Before placement, IsoExec captures the same meta author model in both `.train()` and
`.eval()` modes using `torch.export`. Both semantic programs contribute to its
source identity. The programs must retain the same ordered logical layers,
component calls, state contract, tensor inventory and native PP boundary layouts.
Changing ordinary authored arithmetic between modes is supported. Changing the
component/state/storage structure between modes produces a `ShardError` naming
the required compatible contract.

The trainer compiles both programs over one placed storage owner. `.eval()` and
`.train()` select the captured program; they preserve Parameter objects, native
DDP/optimizer ownership, source identity, weight version and storage generation.
Target program references and OP bindings are checked before the mode changes.
Capture/input guards remain active. The adapter never captures a live placed
model or uses a mode switch to accept stale storage.

SkyRL prescore calls its existing model wrapper's `.eval()`. This read-only path
requires readable model weights and an installed native binding, but allows
native gradients and optimizer state to remain offloaded after a model-only
onload. Training execution still requires those resources to be restored.
Weight update and residency transactions retain their existing execution fences.


<a id="skyrl-native-weight-sessions-and-committed-versions"></a>

### Native weight sessions and committed versions

The extractor declares whether its stream is checkpoint-format. The default
framework extractor requests native checkpoint loading; IsoExec's canonical
extractor requests raw in-place loading. The declaration travels through the
existing IPC/NCCL sender to the receiver.

For vLLM 0.28, the raw-session adapter delegates the actual native Worker's
start/finish bookkeeping and reuses the existing native transfer engine. Its
explicit raw-engine facade follows the native in-place lifecycle contract;
checkpoint layerwise reload is not invoked because it replaces parameter
storage. Native target reset, IPC importer lifetime and model-runner finalization
remain part of the native session. Unsupported checkpoint-format, draft, RDT,
quantized and LoRA sessions are refused before mutation.

After every physical worker finishes, its post-wake RPC revalidates local
storage, records the applied digest/version and returns READY. The vLLM 0.28
EngineCore wrapper requires all native worker replies to identify the same
source version, calls `set_weight_version()`, and validates
`get_weight_version()` in that same RPC. SkyRL waits for every server response
before resume. A local or native version publication failure prevents success.
The framework's prefix-cache salt counter is auxiliary; it does not produce or
replace the committed source version.

vLLM executes its evaluation program while attesting the same pair of semantic
profiles as the trainer. Ambient grad/inference-dependent author branches are
not additional public profiles; native numerical qualification remains required
for the admitted authored model and execution environment.

See [logical layer mapping and actionable PP errors](#architecture-automatic-inference-with-an-annotation-only-when-needed)
for automatic inference and the optional `pipeline_layers()` method. Launchers
and GPU receipts belong in Git-excluded local folders.


<a id="phase-8-native-qualification-and-limits"></a>

## Phase 8: native qualification and limits

The completed snapshot is documented under [native qualification](#skyrl-native-qualification-snapshot).
It covers Qwen3.5-0.8B at PP1 and PP2 on H100, each with W0 comparison,
exactly one real optimizer update, W1 sync/readback/ACK and W1 comparison.
Resolved configurations, full-vocabulary nonzero outputs, negative controls,
native engagement and independent artifact verification remain required.
W0 and all 362 finalized gradients, W0/W1 logits and full logprobs matched across
APIs; W1 retained 9/6 one-ULP BF16 element differences. The full-weight bitwise
gate remains false, and the specific native norm/reduction-order cause is not
independently replayed. PP4, larger models and MTP-on remain unverified.


<a id="architecture-automatic-inference-with-an-annotation-only-when-needed"></a>

## Logical pipeline layers

Automatic inference, with an annotation only when needed.

This section is the detailed `pipeline_layers()` and partition-diagnostic contract.

First attempt to identify one unambiguous ordered sequence of logical decoder layers from module structure and the actual captured dataflow. A `ModuleList` is a candidate, not proof of execution order or layer membership. Never infer layer semantics from model-family names or attribute names such as `layers` or `blocks` alone.

When inference is ambiguous, request this optional method on the authored model:

```python
def pipeline_layers(self):
    return self.blocks
```

An explicit list or tuple of existing module references is also valid:

```python
def pipeline_layers(self):
    return [self.foo, self.bar, self.baz]
```

`pipeline_layers()` optionally provides the ordered `nn.Module` instances corresponding to the logical decoder layers used by the native runtime's PP layer assignment. It is a logical-layer mapping, not a stage definition. Models with an unambiguous inferred mapping need no annotation. If the method is present, validate and use that explicit mapping.

For the first version, each logical layer must have one enclosing `nn.Module`. Group entries such as `(self.attn0, self.mlp0)` are deferred; an author can instead place that computation in a `Block` module. Return the existing modules registered under the model, not newly constructed modules or strings.

The author does not specify PP size, rank ownership, stage numbers, embedding/head ownership, stage inputs/outputs, or FX node ranges in this method. IsoExec derives executable stages and live values across each boundary from the graph and the native runtime's assignment, and rejects a requested partition it cannot implement correctly. Inter-layer operations such as `x * 0.5` remain part of the actual graph and must not be dropped.

Required validation:

- Resolve references to the current model tree and preserve their correspondence to the captured invocation scopes. The declared layer order must agree with the actual dataflow.
- Reject unregistered references, ambiguous module aliases, overlapping layer scopes, and mappings that cannot be reconciled with the graph. A declaration does not bypass dependency, state, layout, or backend-capability checks.
- Preserve the existing requirement of one invocation of each executed stateful module per root `forward()`. Keep the terminal `LMHead` as the authored logits output.
- Derive stage I/O from live graph values, including multiple activation tensors where required. Persistent runtime cache ownership remains a separate state contract; a layer declaration does not implicitly authorize cache sharing or migration.
- Keep an internal reference-to-FQN mapping. FQNs are stable within a captured model identity; renaming a module changes its FQN and may require corresponding weight-mapping changes. Module references avoid a second author-maintained list of string paths, not all rename consequences.

When no unique mapping can be inferred and no annotation is supplied, report:

```text
Cannot uniquely infer PP logical layers.
Provide Model.pipeline_layers() returning the ordered existing nn.Module instances.
```

These are the implemented logical-layer author requirements.
