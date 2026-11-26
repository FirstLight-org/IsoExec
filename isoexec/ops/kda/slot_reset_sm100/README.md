# KDA slot initialization candidate

`reset_fresh_slots` clears the convolution window and complete byte record for
each live decode row at logical position zero. A zeroed record is exactly the
existing `glm_sm100.ops.KdaState` initial representation. The kernel performs
memory stores only; KDA projections, convolution, recurrence and output
projection remain the existing programs.

Slot and position values are GPU inputs. The launch is fixed by the batch shape
and supports replay with different live slots and positions. Null and negative
slots are untouched. Duplicate live slots and out-of-range owners trigger a
device assertion before their own block writes; another block may already have
reset its valid slot. The invalid launch poisons its CUDA context and cannot
produce a usable result.
Convolution and record strides may include page padding. One block owns each
request; no atomics or floating-point tensor arithmetic are used. The compiler
may lower integer address division through floating-point reciprocal/conversion
instructions; these operate on indices rather than state values.

This package is deliberately separate from the legacy KDA `glm_sm100/csrc`
directory. It is not connected to the serving adapter yet. Do not remove the
adapter's graph refusal until GPU and SASS gates pass. The four memory-only cases in `tests/test_reset_gpu.py` and two composed cases in
`runtimes/vllm/flash/tests/test_kda_graph_reset_gpu.py` cover
strided storage/gaps, int32/int64 indices, invalid ownership in isolated CUDA
contexts, and captured module/engine decode against clean full-sequence prefill
while changing slots, positions and inputs. The coordinator compiled the
candidate offline and found no atomic/reduction instructions in its reset
kernels. Runtime correctness still awaits the leased GPU suite.

## Cleanup-preserved development conclusion

README labels initialization kernel candidate; offline SASS check found no atomics/reductions. Keep status and isolated test, avoid crediting automatic production selection. This records the historical experiment conclusion while standalone runs, logs and tuning scripts are removed. Existing numerical contract and measured tables above remain the source of detailed conditions; production selection is governed by the current resolver/model route.
