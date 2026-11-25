"""Check runtime imports, module layering, refusal fields, and tensor ownership. AST scans preserve
import purity; runtime-import exemptions must match live violations and cannot cover core, nn,
or schema.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from dataclasses import dataclass
from typing import Iterable

# Roots scanned, relative to the isoexec package directory.
SCANNED_ROOTS = ("ops", "core", "nn", "schema")

# Include the top-level package files: plugin imports must remain free of runtime and CUDA
# initialization.
SCANNED_FILES = ("__init__.py", "config.py")

# Import layering applies to every shipping module, including runtime adapters.
LAYER_ROOTS = (
    *SCANNED_ROOTS, "checks", "compiler", "runtimes", "models", "integrations", "state", "storage", "sync", "weights",
)
LAYER_FILES = ("__init__.py", "config.py")

# Roots that must stay at zero violations and zero exemptions, forever.
ZERO_EXEMPTION_ROOTS = ("core", "nn", "schema")

# Enforce nn -> ops -> core -> schema dependencies and forbid nn -> runtimes.
FORBIDDEN_LAYER_EDGES: tuple[tuple[str, str], ...] = (
    ("core", "ops"), ("core", "nn"),
    ("ops", "nn"),
    ("schema", "core"), ("schema", "ops"), ("schema", "nn"),
    ("nn", "runtimes"),
)

# Production code may not import isoexec.tests; tests may share helpers from it.
PRODUCTION_TEST_IMPORT_ROOT = "tests"

# Kept as the general form: ``(importer root, imported root)`` pairs that fire on production source
# only. Every scanned root is forbidden to reach test code.
PRODUCTION_ONLY_LAYER_EDGES: tuple[tuple[str, str], ...] = tuple(
    (root, PRODUCTION_TEST_IMPORT_ROOT) for root in (*LAYER_ROOTS, *LAYER_FILES)
)


_SCHEMA_REASON = (
    "schema/ is the BOTTOM tier: it is the vocabulary a config is written in, and every tier above\n"
    "    reads it. A schema/ module that imports {tier} inverts that and makes the type a model is\n"
    "    described with depend on the machinery that executes it. This fires on tests too, on\n"
    "    purpose: a proof that needs a composition is a proof about the tier that composes, and it\n"
    "    belongs in that tier's test folder."
)

LAYER_EDGE_REASONS: dict[tuple[str, str], str] = {
    **{
        (root, PRODUCTION_TEST_IMPORT_ROOT): (
            "isoexec/tests/ holds test code and shared test helpers, not a tier: production source\n"
            "    importing it makes a test helper load-bearing. What production and tests both need lives\n"
            "    in the tier that owns it -- schema/ for declarations, core/ for the torch helpers. A test\n"
            "    may import it, and that is why this rule does not fire on test files."
        )
        for root in (*LAYER_ROOTS, *LAYER_FILES)
    },
    ("core", "ops"): (
        "core/ is the shared-machinery tier and ops/ is the op-implementation and registry tier;\n"
        "    the dependency runs ops -> core only. There is no exemption for this and there will not\n"
        "    be one. If core/ needs it, it is not op-specific: move THAT into core/ and import it\n"
        "    from ops/ instead."
    ),
    ("ops", "nn"): (
        "nn/ is the component tier and sits ABOVE ops/: a component reaches for a kernel, never the\n"
        "    reverse. An op package that needs something from a component takes it duck-typed, the way\n"
        "    ops/mm/packed_proj.py and ops/moe/shared/packed_experts.py name a hook and nothing else."
    ),
    ("nn", "runtimes"): (
        "nn/ is the component tier and runtimes/ is the host tier: a component cannot reach a host, or\n"
        "    it could only be built by the host that owns the right table. What a component needs from a\n"
        "    host arrives as a Placement it is handed, or in a slot the host assigns."
    ),
    ("core", "nn"): (
        "nn/ is the component tier and sits ABOVE core/, for the same reason it sits above ops/.\n"
        "    core/ is what nn/ builds FROM; a core/ module that reaches up into a component is the\n"
        "    cycle the tier split exists to avoid. Take what you need duck-typed, or move it down\n"
        "    into core/ where nn/ can import it."
    ),
    ("schema", "core"): _SCHEMA_REASON.format(tier="core/"),
    ("schema", "ops"): _SCHEMA_REASON.format(tier="ops/"),
    ("schema", "nn"): _SCHEMA_REASON.format(tier="nn/"),
}

# The dotted name of the isoexec package, for recognising an ABSOLUTE import of one of its roots.
PACKAGE_NAME = "isoexec"

# Top-level packages that count as "a runtime".
RUNTIME_PACKAGES = ("vllm", "megatron")

CATEGORIES = {
    # Debt. These must shrink. Each is a decision the design note has already taken.
    "a": "numerics we bind and do not own -- VENDOR (kernel-ownership.md §2a, §5 steps 3-5)",
    "b": "bridge-era monkeypatch target, deletion candidate (kernel-ownership.md §2b, §5 step 2)",
    # Permanent. These are allowed to stay, but still have to be listed and justified.
    "c": "process/memory/registration plumbing, not numerics -- LEAVE (kernel-ownership.md §2c)",
    "gate": "test that binds upstream on purpose, to prove our numerics equal upstream's bitwise",
}
# The categories that are DEBT. Every other category in CATEGORIES is permanent; the summary
# reads the complement rather than a second tuple that could disagree with this one.
DEBT_CATEGORIES = ("a", "b")

_HERE = os.path.dirname(os.path.abspath(__file__))
ISOEXEC_DIR = os.path.dirname(_HERE)
EXEMPTIONS_PATH = os.path.join(_HERE, "runtime_import_exemptions.json")
# Fixed location inside the package; used in messages so they read the same from any working dir.
LEDGER_LABEL = "isoexec/checks/runtime_import_exemptions.json"


@dataclass(frozen=True)
class Violation:
    """One runtime import, normalised to the dotted thing it takes out of the runtime."""

    path: str  # posix, relative to ISOEXEC_DIR
    target: str  # e.g. "vllm.model_executor.determinism.batch_invariant"
    lineno: int
    statement: str

    def __str__(self) -> str:
        return f"{self.path}:{self.lineno}: {self.target}    [{self.statement}]"


@dataclass(frozen=True)
class Exemption:
    path: str
    prefix: str
    category: str
    reason: str

    def covers(self, v: Violation) -> bool:
        if v.path != self.path:
            return False
        return v.target == self.prefix or v.target.startswith(self.prefix + ".")


@dataclass(frozen=True)
class LayerViolation:
    """One import that crosses a forbidden module-layering edge."""

    path: str  # posix, relative to ISOEXEC_DIR -- the importing file
    importer_root: str  # e.g. "core"
    imported_root: str  # e.g. "ops"
    target: str  # the resolved dotted name, relative to the isoexec package
    lineno: int
    statement: str

    def __str__(self) -> str:
        return f"{self.path}:{self.lineno}: {self.importer_root}/ -> {self.target}    [{self.statement}]"


def _is_runtime(dotted: str) -> bool:
    return dotted.split(".", 1)[0] in RUNTIME_PACKAGES


def _runtime_nodes(tree):
    """Walk executable syntax; TYPE_CHECKING imports describe interfaces but never bind kernels."""
    yield tree
    if isinstance(tree, ast.If) and (
        isinstance(tree.test, ast.Name) and tree.test.id == "TYPE_CHECKING"
        or isinstance(tree.test, ast.Attribute) and isinstance(tree.test.value, ast.Name)
        and tree.test.value.id == "typing" and tree.test.attr == "TYPE_CHECKING"
    ):
        for node in tree.orelse:
            yield from _runtime_nodes(node)
        return
    for node in ast.iter_child_nodes(tree):
        yield from _runtime_nodes(node)


def scan_file(abspath: str, relpath: str) -> list[Violation]:
    """Find runtime imports at any scope, including resolvable dynamic-import prefixes. Normalize
    from-imports to their imported targets.
    """
    with open(abspath, "r", encoding="utf-8") as fh:
        src = fh.read()
    try:
        tree = ast.parse(src, filename=abspath)
    except SyntaxError as exc:
        raise RuntimeError(f"{relpath}: cannot parse: {exc}") from exc

    out: list[Violation] = []
    for node in _runtime_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_runtime(alias.name):
                    out.append(Violation(relpath, alias.name, node.lineno, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            module = node.module or ""
            if not _is_runtime(module):
                continue
            names = [a.name for a in node.names]
            stmt = f"from {module} import {', '.join(names)}"
            for name in names:
                target = module if name == "*" else f"{module}.{name}"
                out.append(Violation(relpath, target, node.lineno, stmt))
        elif isinstance(node, ast.Call):
            dyn = _dynamic_import_target(node)
            if dyn is not None and _is_runtime(dyn):
                out.append(Violation(relpath, dyn, node.lineno, f"importlib.import_module({dyn!r}...)"))
    return out


_DYNAMIC_IMPORT_FUNCS = ("import_module", "__import__")


def _dynamic_import_target(node: ast.Call) -> str | None:
    """Return a dynamic import's literal module prefix, or None when it cannot be resolved."""
    func = node.func
    fname = func.attr if isinstance(func, ast.Attribute) else (func.id if isinstance(func, ast.Name) else None)
    if fname not in _DYNAMIC_IMPORT_FUNCS or not node.args:
        return None
    arg = node.args[0]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
    if isinstance(arg, ast.JoinedStr) and arg.values:
        head = arg.values[0]
        if isinstance(head, ast.Constant) and isinstance(head.value, str):
            return head.value.rstrip(".") or None
    return None


def _package_parts(relpath: str) -> list[str]:
    """The package a file's relative imports resolve against, as parts under the isoexec package.

    ``nn/moe/layer.py`` -> ``["nn", "moe"]``. For ``__init__.py`` that is the package itself,
    which is the same directory, so the dirname works for both module and package files.
    """
    return [p for p in os.path.dirname(relpath).split("/") if p]


def _resolved_root(relpath: str, level: int, module: str) -> str | None:
    """Resolve an absolute or relative IsoExec import to its top-level package."""
    if level:
        base = _package_parts(relpath)
        # level 1 is the containing package; each extra level strips one more.
        if level - 1 > len(base):  # climbs above the isoexec package: cannot name a root
            return None
        parts = base[: len(base) - (level - 1)] + [p for p in module.split(".") if p]
    else:
        parts = [p for p in module.split(".") if p]
        if PACKAGE_NAME not in parts:
            return None
        parts = parts[parts.index(PACKAGE_NAME) + 1 :]
    return parts[0] if parts else None


def scan_file_layering(abspath: str, relpath: str) -> list[LayerViolation]:
    """Every forbidden cross-root import in one file, top-level and function-local alike.

    Relative and absolute forms both count, and so does ``importlib.import_module`` -- a layering
    rule an inner function or a string can walk around is not a rule.
    """
    importer_root = relpath.split("/", 1)[0]
    parts = relpath.split("/")
    is_test = "tests" in parts or parts[-1].startswith("test_")
    edges = FORBIDDEN_LAYER_EDGES if is_test else FORBIDDEN_LAYER_EDGES + PRODUCTION_ONLY_LAYER_EDGES
    forbidden = {imported for importer, imported in edges if importer == importer_root}
    if not forbidden:
        return []

    with open(abspath, "r", encoding="utf-8") as fh:
        src = fh.read()
    try:
        tree = ast.parse(src, filename=abspath)
    except SyntaxError as exc:
        raise RuntimeError(f"{relpath}: cannot parse: {exc}") from exc

    out: list[LayerViolation] = []

    def record(root: str | None, target: str, lineno: int, stmt: str) -> None:
        if root in forbidden:
            out.append(LayerViolation(relpath, importer_root, root, target, lineno, stmt))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                record(_resolved_root(relpath, 0, alias.name), alias.name, node.lineno, f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [a.name for a in node.names]
            dots = "." * node.level
            stmt = f"from {dots}{module} import {', '.join(names)}"
            root = _resolved_root(relpath, node.level, module)
            record(root, f"{dots}{module}", node.lineno, stmt)
        elif isinstance(node, ast.Call):
            dyn = _dynamic_import_target(node)
            if dyn is not None:
                record(_resolved_root(relpath, 0, dyn), dyn, node.lineno, f"importlib.import_module({dyn!r}...)")
    return out


def _walk_py(isoexec_dir: str, roots: Iterable[str], files: Iterable[str] = SCANNED_FILES):
    """(abspath, relpath) for every .py file under ``roots`` plus each named file, in a stable order."""
    for name in sorted(files):
        abspath = os.path.join(isoexec_dir, name)
        if os.path.isfile(abspath):
            yield abspath, name
    for root in roots:
        base = os.path.join(isoexec_dir, root)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
            for filename in sorted(filenames):
                if not filename.endswith(".py"):
                    continue
                abspath = os.path.join(dirpath, filename)
                yield abspath, os.path.relpath(abspath, isoexec_dir).replace(os.sep, "/")


def scan_layering(
    isoexec_dir: str | None = None,
    roots: Iterable[str] = LAYER_ROOTS,
    files: Iterable[str] = LAYER_FILES,
) -> list[LayerViolation]:
    """Every forbidden cross-root import under ``roots``. Deduplicated per (file, target)."""
    isoexec_dir = isoexec_dir or ISOEXEC_DIR
    found: list[LayerViolation] = []
    for abspath, relpath in _walk_py(isoexec_dir, roots, files):
        found.extend(scan_file_layering(abspath, relpath))
    seen: dict[tuple[str, str], LayerViolation] = {}
    for v in found:
        seen.setdefault((v.path, v.target), v)
    return sorted(seen.values(), key=lambda v: (v.path, v.target))


def scan(
    isoexec_dir: str | None = None,
    roots: Iterable[str] = SCANNED_ROOTS,
    files: Iterable[str] = SCANNED_FILES,
) -> list[Violation]:
    # Resolved at call time, not bound as a default, so tests can point the scan at a sandbox tree.
    isoexec_dir = isoexec_dir or ISOEXEC_DIR
    found: list[Violation] = []
    for abspath, relpath in _walk_py(isoexec_dir, roots, files):
        found.extend(scan_file(abspath, relpath))
    # Deduplicate: the same target imported in three functions of one file is one debt, not three.
    seen: dict[tuple[str, str], Violation] = {}
    for v in found:
        seen.setdefault((v.path, v.target), v)
    return sorted(seen.values(), key=lambda v: (v.path, v.target))


def load_exemptions(path: str | None = None) -> list[Exemption]:
    path = path or EXEMPTIONS_PATH
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    out = []
    for i, entry in enumerate(raw["exemptions"]):
        missing = {"path", "prefix", "category", "reason"} - set(entry)
        if missing:
            raise ValueError(f"exemption #{i} is missing {sorted(missing)}: {entry}")
        if entry["category"] not in CATEGORIES:
            raise ValueError(
                f"exemption #{i} has unknown category {entry['category']!r}; expected one of {sorted(CATEGORIES)}"
            )
        if not entry["reason"].strip():
            raise ValueError(f"exemption #{i} ({entry['path']}) has an empty reason; every entry must say why")
        out.append(Exemption(entry["path"], entry["prefix"], entry["category"], entry["reason"]))
    return out


def check(isoexec_dir: str | None = None, exemptions_path: str | None = None) -> list[str]:
    """Return a list of failure messages. Empty list == clean."""
    isoexec_dir = isoexec_dir or ISOEXEC_DIR
    exemptions_path = exemptions_path or EXEMPTIONS_PATH
    violations = scan(isoexec_dir)
    exemptions = load_exemptions(exemptions_path)
    failures: list[str] = []

    for ex in exemptions:
        root = ex.path.split("/", 1)[0]
        if root in ZERO_EXEMPTION_ROOTS:
            failures.append(
                f"FORBIDDEN EXEMPTION: {ex.path} is under {root}/, which is enforced at zero runtime "
                f"imports with no exemptions. Do not add one -- remove the import instead."
            )

    matched: dict[Exemption, int] = {ex: 0 for ex in exemptions}
    for v in violations:
        covering = [ex for ex in exemptions if ex.covers(v)]
        for ex in covering:
            matched[ex] += 1
        if not covering:
            failures.append(
                f"NEW RUNTIME IMPORT: {v}\n"
                f"    a scanned tier must not bind a runtime. If this import is numerics, it has to be\n"
                f"    vendored into our tree (kernel-ownership.md §1). If it is genuinely plumbing, add an\n"
                f"    entry to {LEDGER_LABEL} with category 'c' and a reason."
            )

    for v in scan_layering(isoexec_dir):
        reason = LAYER_EDGE_REASONS.get(
            (v.importer_root, v.imported_root),
            f"{v.importer_root}/ may not import {v.imported_root}/, and the edge carries no exemption mechanism.",
        )
        failures.append(f"FORBIDDEN LAYERING IMPORT: {v}\n    {reason}")

    for ex, hits in matched.items():
        if hits == 0:
            failures.append(
                f"STALE EXEMPTION: {ex.path} :: {ex.prefix} (category {ex.category}) no longer matches any import.\n"
                f"    The import is gone -- good. Delete this entry so the ledger keeps shrinking.\n"
                f"    Reason it carried: {ex.reason}"
            )

    failures.extend(tensor_ownership_violations(isoexec_dir))
    return failures


# Refusal fields.

# Scan checks and runtime adapters that read resolved engine configuration.
REFUSAL_ROOTS = ("ops", "sync", "runtimes", "checks")

# Tracked engine-config aliases; defaulted getattr would conceal renamed fields.
ENGINE_CONFIG_NAMES = frozenset(
    {
        "vllm_config",
        "_vllm_config",
        "kv_cache_config",
        "parallel_config",
        "model_config",
        "cache_config",
        "scheduler_config",
        "compilation_config",
        "speculative_config",
    }
)


@dataclass(frozen=True)
class FieldRead:
    """One ``<config>.<field>`` read, with where it was written."""

    path: str
    line: int
    obj: str
    field: str
    defaulted: bool  # a three-argument getattr: the read cannot fail, so a rename is invisible
    config_type: str = ""

    def __str__(self) -> str:
        how = "getattr(..., default)" if self.defaulted else "attribute read"
        return f"{self.path}:{self.line}: {self.obj}.{self.field} ({how})"


def _getattr_call(node: ast.Call) -> tuple[str, str, bool] | None:
    """``(obj, field, defaulted)`` for ``getattr(obj, "field"[, default])``, else ``None``."""
    if not (isinstance(node.func, ast.Name) and node.func.id == "getattr"):
        return None
    if len(node.args) < 2 or not isinstance(node.args[1], ast.Constant) or not isinstance(node.args[1].value, str):
        return None
    target = node.args[0]
    name = target.id if isinstance(target, ast.Name) else (target.attr if isinstance(target, ast.Attribute) else None)
    if name is None:
        return None
    return name, node.args[1].value, len(node.args) > 2


CONFIG_TYPES = frozenset({"TransformerConfig", "OptimizerConfig", "DistributedDataParallelConfig", "VllmConfig"})


def _annotation_name(node: ast.expr | None) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.rsplit(".", 1)[-1]
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def scan_field_reads(abspath: str, relpath: str) -> list[FieldRead]:
    """Read engine names and explicitly typed native configs, without importing the scanned file.

    Native config variable names are not reserved: ``cfg`` also holds our model schema. Type
    annotations identify native configs, with a separate binding scope for each function.
    """
    with open(abspath, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=abspath)
    out: list[FieldRead] = []

    class Reads(ast.NodeVisitor):
        def __init__(self):
            self.bindings: dict[str, str] = {}

        def visit_FunctionDef(self, node):
            previous = self.bindings
            self.bindings = dict(previous)
            for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                kind = _annotation_name(arg.annotation)
                if kind:
                    self.bindings[arg.arg] = kind if kind in CONFIG_TYPES else ""
            self.generic_visit(node)
            self.bindings = previous

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_AnnAssign(self, node):
            kind = _annotation_name(node.annotation)
            self.bindings[ast.unparse(node.target)] = kind if kind in CONFIG_TYPES else ""
            self.generic_visit(node)

        def record(self, target, field, line, defaulted):
            name = ast.unparse(target)
            terminal = _annotation_name(target)
            kind = self.bindings.get(name)
            if kind is None:
                kind = terminal if terminal in ENGINE_CONFIG_NAMES else ""
            if kind:
                out.append(FieldRead(relpath, line, name, field, defaulted, kind))

        def visit_Call(self, node):
            found = _getattr_call(node)
            if found is not None:
                self.record(node.args[0], found[1], node.lineno, found[2])
            self.generic_visit(node)

        def visit_Attribute(self, node):
            if isinstance(node.ctx, ast.Load):
                self.record(node.value, node.attr, node.lineno, False)
            self.generic_visit(node)

    Reads().visit(tree)
    return out


def refusal_field_violations(isoexec_dir: str | None = None, roots: Iterable[str] = REFUSAL_ROOTS) -> list[str]:
    """Reject defaulted engine-config reads and fields absent from the pinned configuration
    classes. Report unavailable classes as unverifiable rather than clean.
    """
    isoexec_dir = isoexec_dir or ISOEXEC_DIR
    reads: list[FieldRead] = []
    for abspath, relpath in _walk_py(isoexec_dir, roots):
        reads.extend(scan_field_reads(abspath, relpath))
    problems = [
        f"DEFAULTED REFUSAL FIELD: {r}\n"
        f"    A three-argument getattr over an engine config cannot fail, so an upstream rename turns\n"
        f"    this refusal into a no-op silently. Read the attribute and let the AttributeError say so."
        for r in reads
        if r.defaulted
    ]
    problems.extend(_unknown_field_problems(reads))
    return problems


def _declared_fields(cls) -> frozenset[str]:
    # Required dataclass fields have annotations but no class-level default, so dir is insufficient.
    return frozenset(dir(cls)) | frozenset(name for base in cls.__mro__ for name in vars(base).get("__annotations__", {}))


def _engine_config_fields() -> dict[str, frozenset] | None:
    """``{local name: declared fields}`` from the PINNED engine wheels, or ``None`` when absent."""
    known: dict[str, frozenset] = {}
    try:
        from vllm import config as vllm_config_mod

        known["vllm_config"] = _declared_fields(vllm_config_mod.VllmConfig)
        for local, cls in (
            ("parallel_config", "ParallelConfig"),
            ("model_config", "ModelConfig"),
            ("cache_config", "CacheConfig"),
            ("scheduler_config", "SchedulerConfig"),
            ("compilation_config", "CompilationConfig"),
            ("speculative_config", "SpeculativeConfig"),
        ):
            attr = getattr(vllm_config_mod, cls, None)
            if attr is not None:
                known[local] = _declared_fields(attr)
    except Exception:  # noqa: BLE001 -- no vLLM wheel in this job
        pass
    if "vllm_config" in known:
        known["VllmConfig"] = known["_vllm_config"] = known["vllm_config"]
    try:
        from vllm.v1.kv_cache_interface import KVCacheConfig

        known["kv_cache_config"] = _declared_fields(KVCacheConfig)
    except ImportError:
        pass
    try:
        from megatron.core.distributed import DistributedDataParallelConfig
        from megatron.core.optimizer import OptimizerConfig
        from megatron.core.transformer.transformer_config import TransformerConfig

        for cls in (TransformerConfig, OptimizerConfig, DistributedDataParallelConfig):
            known[cls.__name__] = _declared_fields(cls)
    except ImportError:
        pass
    return known or None


def _unknown_field_problems(reads: list[FieldRead]) -> list[str]:
    known = _engine_config_fields()
    if known is None:
        return [
            "UNVERIFIABLE REFUSAL FIELDS: vllm.config does not import in this process,\n"
            "    so no refusal field could be checked against the pinned engine. This rule runs\n"
            "    in the job that HAS the wheels; a skip here is a failure, not a pass."
        ]
    out: list[str] = []
    for read in reads:
        fields = known.get(read.config_type or read.obj)
        if fields is None or read.field in fields:
            continue
        out.append(
            f"UNKNOWN REFUSAL FIELD: {read}\n"
            f"    The pinned engine config class declares no such field, so this refusal can never fire."
        )
    return out


TENSOR_ROOTS = ("compiler", "nn", "ops", "runtimes")
_TENSOR_FACTORIES = frozenset(
    {
        "empty",
        "zeros",
        "ones",
        "full",
        "tensor",
        "as_tensor",
        "arange",
        "rand",
        "randn",
        "empty_like",
        "zeros_like",
        "ones_like",
        "full_like",
        "cat",
        "stack",
    }
)


def scan_tensor_ownership(abspath: str, relpath: str) -> list[str]:
    """Check allocation metadata and attribute writes on statically identified tensors. Framework
    parameter metadata is exempt; arbitrary return types are not inferred.
    """
    with open(abspath, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=abspath)
    problems = []

    class TensorReads(ast.NodeVisitor):
        def __init__(self):
            self.tensors = set()

        def tensor(self, node):
            if isinstance(node, ast.Name):
                return node.id in self.tensors
            return (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"
                and node.func.attr in _TENSOR_FACTORIES
            )

        def visit_FunctionDef(self, node):
            previous = self.tensors
            self.tensors = set(previous)
            for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                self.tensors.discard(arg.arg)
                if _annotation_name(arg.annotation) == "Tensor":
                    self.tensors.add(arg.arg)
            self.generic_visit(node)
            self.tensors = previous

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_AnnAssign(self, node):
            if isinstance(node.target, ast.Name) and _annotation_name(node.annotation) == "Tensor":
                self.tensors.add(node.target.id)
            self.generic_visit(node)

        def visit_Assign(self, node):
            tensor = self.tensor(node.value)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.tensors.discard(target.id)
                    if tensor:
                        self.tensors.add(target.id)
            self.generic_visit(node)

        def visit_Call(self, node):
            func = node.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "torch":
                if func.attr in {"empty", "zeros", "ones", "full"}:
                    missing = {"dtype", "device"} - {kw.arg for kw in node.keywords}
                    if missing:
                        problems.append(
                            f"IMPLICIT TENSOR ALLOCATION: {relpath}:{node.lineno}: "
                            f"torch.{func.attr} must name {', '.join(sorted(missing))}"
                        )
            if isinstance(func, ast.Name) and func.id == "setattr" and node.args and self.tensor(node.args[0]):
                problems.append(
                    f"TENSOR ATTRIBUTE: {relpath}:{node.lineno}: "
                    "carry metadata in the result or register storage on its owning module"
                )
            self.generic_visit(node)

    TensorReads().visit(tree)
    return problems


def tensor_ownership_violations(isoexec_dir: str | None = None) -> list[str]:
    """Production source only; tests construct intentionally invalid inputs for these checks."""
    problems = []
    for abspath, relpath in _walk_py(isoexec_dir or ISOEXEC_DIR, TENSOR_ROOTS, files=()):
        parts = relpath.split("/")
        if "tests" in parts or parts[-1].startswith("test_"):
            continue
        problems.extend(scan_tensor_ownership(abspath, relpath))
    return problems


def _counts(violations: list[Violation], exemptions: list[Exemption]) -> dict[str, int]:
    per_cat: dict[str, int] = {k: 0 for k in CATEGORIES}
    for v in violations:
        for ex in exemptions:
            if ex.covers(v):
                per_cat[ex.category] += 1
                break
    return per_cat


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--summary", action="store_true", help="print counts per category")
    ap.add_argument("--list", action="store_true", help="print every live violation with its classification")
    args = ap.parse_args(argv)

    violations = scan(ISOEXEC_DIR)
    exemptions = load_exemptions(EXEMPTIONS_PATH)

    if args.list:
        for v in violations:
            cat = next((ex.category for ex in exemptions if ex.covers(v)), "UNCLASSIFIED")
            print(f"[{cat:>4}] {v}")
        for lv in scan_layering(ISOEXEC_DIR):
            print(f"[LAYER] {lv}")

    if args.summary:
        per_cat = _counts(violations, exemptions)
        files = len({v.path for v in violations})
        print(f"runtime imports under {'/, '.join(SCANNED_ROOTS)}/: {len(violations)} in {files} files")
        for cat, n in per_cat.items():
            kind = "DEBT     " if cat in DEBT_CATEGORIES else "permanent"
            print(f"  ({cat:>4}) {kind} {n:>4}  {CATEGORIES[cat]}")
        debt = sum(per_cat[c] for c in DEBT_CATEGORIES)
        print(f"  outstanding debt (a+b): {debt}   -- this number must never go up")
        layering = scan_layering(ISOEXEC_DIR)
        for importer, imported in FORBIDDEN_LAYER_EDGES:
            n = len([v for v in layering if (v.importer_root, v.imported_root) == (importer, imported)])
            print(f"  {importer}/ -> {imported}/ imports: {n}   -- no exemptions exist; this must stay 0")

    failures = check(ISOEXEC_DIR, EXEMPTIONS_PATH)
    if failures:
        print("\nruntime-import lint FAILED:\n", file=sys.stderr)
        for f in failures:
            print(f"  {f}\n", file=sys.stderr)
        print(f"  {len(failures)} problem(s). Ledger: {LEDGER_LABEL}", file=sys.stderr)
        return 1
    if not (args.summary or args.list):
        print(f"runtime-import lint OK: {len(violations)} import(s), all classified; no stale entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
