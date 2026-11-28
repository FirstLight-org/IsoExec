# kda/slot_reset_sm100: graph-safe reset of fresh KDA slots on Blackwell (SM100)

A memory-only kernel that clears the per-slot state of the KDA operators (`kda/glm_sm100`) for every
decode row that starts a sequence: the convolution window and the complete byte record of its slot. A
zeroed record is exactly the initial representation of `glm_sm100.ops.KdaState` (a fresh sequence at
position 0), so a reset slot continues bit for bit as a sequence that was never in the slot before. The
kernel performs stores only; the KDA projections, convolution, recurrence and output projection are the
existing programs and are not touched.

```python
from isoexec.ops.kda.slot_reset_sm100 import reset_fresh_slots, reset_fresh_slots_layers
reset_fresh_slots(conv, records, slots, positions, null_slot=0)                 # one layer, in place
reset_fresh_slots_layers(convs, records_per_layer, slots, positions, null_slot=0)  # every layer, one launch
```

* `conv`: the convolution windows, bf16 `[slots, D, C]`; `records`: the byte records, uint8 `[slots, H, B]`.
  Both are slot-major with the same slot capacity; any strides, including page padding and gaps.
* `slots`, `positions`: contiguous int32 or int64 vectors of one entry per decode row, on the state's
  device. A row whose position is 0 has its slot cleared; every other row is left as it is.
* Null rows (`slot == null_slot`) and negative slots are padding and are not touched.

Slot ids and positions are GPU inputs. The launch is fixed by the batch shape (one block of 256 threads
per row, and per layer for `reset_fresh_slots_layers`), so a captured CUDA graph replays with different
live slots and positions. Neither state tensor is reallocated. One block owns each row's slot; there are
no atomics and no floating-point arithmetic on state values (the compiler may lower integer address
division through floating-point reciprocal and conversion instructions, which act on indices only).

## Limits

* Each live row must own a distinct slot below the capacity. A duplicate live slot or an out-of-range slot
  triggers a device assertion before that block writes; another block may already have reset its own
  valid slot. The assertion poisons the CUDA context, and the call has no usable result.
* `reset_fresh_slots_layers` takes at most 64 layers, all of one state geometry (sizes and strides) and
  sharing the step's slots and positions; it refuses mixed geometries. Its stores are those of one
  `reset_fresh_slots` per layer.
* One CUDA device; bf16 convolution windows and byte records only.
* The package is not connected to a serving adapter in this tree: nothing calls it but its tests.

## Evidence

| claim | test |
|---|---|
| strided storage with gaps, int32 and int64 indices, continuation rows and null padding untouched | `tests/test_reset_gpu.py` |
| an invalid live ownership refuses, in an isolated CUDA context | `tests/test_reset_gpu.py` |
| one multi-layer launch == one launch per layer; mixed geometries are refused | `tests/test_reset_gpu.py` |
| a reset slot is a fresh sequence at position 0 for both KDA reducers | `isoexec/ops/kda/tests/test_schedule_invariance_gpu.py` |
