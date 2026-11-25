"""The phased transport over a private workspace, serialized by its owning execution runtime.

One fused launch per chunk (``tree_fused_kernel``: slot tickets and call credits, one-shot or two-shot by
size). Supported graph replay MUST be inside ``execution_scope``, including an external manager's replay;
a raw ``CUDAGraph.replay`` outside that runtime is unsupported. Capture is a quiescent lifecycle operation;
it may temporarily use capture streams.
"""
from contextlib import contextmanager
import threading

class ExecutionOwnership:
    def __init__(self, current, capturing, synchronize):
        self.current, self.capturing, self.synchronize = current, capturing, synchronize
        self.lock = threading.Lock()
        self.owner = None
        self.depth = 0
        self.capture = False
        self.runtime_stream = None
        self.capture_stream = None

    def validate(self):
        if self.owner != threading.get_ident() or not self.depth:
            raise RuntimeError('phased transport requires an owning runtime execution scope')
        stream = self.current()
        if self.capture:
            if stream != self.capture_stream:
                if self.capturing():
                    raise RuntimeError('capture stream must be warmed before capture')
                self.synchronize()
                self.capture_stream = stream
        elif self.capturing() or stream != self.runtime_stream:
            raise RuntimeError('phased runtime CUDA stream changed or capture lacks lifecycle scope')

    @contextmanager
    def scope(self, *, capture=False):
        thread = threading.get_ident()
        nested = self.owner == thread
        if nested:
            if capture or not self.capture:
                raise RuntimeError('nested phased execution is only allowed inside capture lifecycle')
            self.depth += 1
            try:
                self.validate()
                yield
            finally:
                self.depth -= 1
            return
        if not self.lock.acquire(False):
            raise RuntimeError('concurrent phased runtime submission refused')
        try:
            if self.capturing():
                raise RuntimeError('enter phased execution scope before CUDA capture')
            self.owner, self.depth, self.capture = thread, 1, bool(capture)
            if capture:
                self.synchronize()
                self.capture_stream = self.current()
            elif self.runtime_stream is None:
                self.runtime_stream = self.current()
            self.validate()
            try:
                yield
            finally:
                if capture:
                    self.synchronize()
        finally:
            self.owner, self.depth, self.capture = None, 0, False
            self.capture_stream = None
            self.lock.release()

def aligned_root(root):
    """Zero-pad independent columns to whole float4 groups."""
    import torch.nn.functional as F
    flat = root.contiguous().view(-1)
    padding = (-flat.numel()) % 4
    return F.pad(flat, (0, padding), value=0.0) if padding else flat

#: One-shot while (W - 1) * n <= this many elements (and n fits one sender slot), two-shot above.
FUSED_ONESHOT_ELEMS = 512 << 10
#: CTAs per launch: the SM count (tiles per item 1 while that covers the payload, else 4). The grid never
#: changes bits (same tree, same operands, any grid).
TILE = 1024


def fused_plan(n, world, lslot, lycap, lcap, max_grid, oneshot_elems, two_shot_chunk):
    """[(start, stop, two_shot, grid, unroll)] of one fused call: a pure function of (n, W, workspace geometry, tunables),
    so every rank derives the same launches."""
    plan = []
    two = not ((world - 1) * n <= oneshot_elems and n <= lslot)
    start = 0
    while start < n:
        rest = n - start
        unroll = 1 if -(-rest // TILE) <= max_grid else 4
        item = unroll * TILE
        grid = min(max_grid, -(-rest // item))
        if two:
            grid = max(1, min(grid, lslot // item))    # one round of `grid` items must fit a compact sender slot (W = 8: 128)
            step = two_shot_chunk(lcap, lycap, world, grid, unroll)
            if step <= 0:
                raise ValueError('fused two-shot: workspace too small for one item round')
        else:
            step = lslot
        stop = min(n, start + step)
        if stop - start < rest:   # a full chunk: grid / unroll already sized for >= its item count
            grid = min(grid, -(-(stop - start) // item))
        plan.append((start, stop, two, grid, unroll))
        start = stop
    return plan


class PhasedTransport:
    def __init__(self, workspace, extension):
        import torch
        self.ws, self.extension = workspace, extension
        # Caller supplies canonical device explicitly through workspace sync state.
        device = workspace.sync.state.device
        self.sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.oneshot_elems = FUSED_ONESHOT_ELEMS
        self.lslot = int(extension.tree_slot(workspace.lcap, max(workspace.world, 1)))
        self._plans = {}
        self.ownership = ExecutionOwnership(
            lambda: (device.index, torch.cuda.current_stream(device).cuda_stream),
            torch.cuda.is_current_stream_capturing,
            lambda: torch.cuda.synchronize(device))

    def signature(self):
        """Everything the launch sequence depends on besides the payload size (must agree across ranks)."""
        return ("fused", self.sms, self.oneshot_elems, self.ws.lcap, self.ws.lycap, self.ws.world)

    def execution_scope(self, *, capture=False):
        return self.ownership.scope(capture=capture)

    def plan(self, n):
        p = self._plans.get(n)
        if p is None:
            ws = self.ws
            p = self._plans[n] = fused_plan(n, ws.world, self.lslot, ws.lycap, ws.lcap, self.sms, self.oneshot_elems,
                                            self.extension.fused_two_shot_chunk)
        return p

    def __call__(self, root):
        import torch
        self.ownership.validate()
        ws = self.ws
        if not root.is_cuda or root.dtype != torch.float32:
            raise TypeError('phased transport requires CUDA FP32 subtree roots')
        ri = aligned_root(root)
        ro = torch.empty_like(ri, dtype=torch.bfloat16)
        for start, stop, two, grid, unroll in self.plan(ri.numel()):
            self.extension.fused_entry(ri[start:stop], ro[start:stop], ws.lbox_ptrs, ws.lcap, ws.lybox_ptrs, ws.lycap,
                                       ws.sync.flag_ptrs, ws.sync.state, ws.rank, int(two), grid, unroll)
        return ro[:root.numel()].view(root.shape)
