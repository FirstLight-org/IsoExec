"""The scope proof for ``bwd_compile._BACKWARD_ONLY_SITES``.

``bwd_compile.py`` claims this file exists and asserts each table entry against its target's call
sites: *"This table is the scope proof; ``tests/test_bwd_compile_cpu.py`` asserts each entry against
the target's call sites, so adding a site without extending that test fails."*
It did not exist. Written 2026-09-05 (STATUS section 24.8) to make the claim true rather than to
delete it, because the claim is load-bearing: ``call_region`` reassociates backward reductions, which
is legal ONLY because every registered site is unreachable from a forward. Nothing but this file
checks that.

The proof is mechanical, not textual. ``_callers_within`` builds a static call graph from the AST and
walks it UPWARD from each ``call_region`` call site; the walk must terminate only at
``autograd.Function.backward`` staticmethods. A grep for "backward" in the filename would prove
nothing -- both call sites sit in module-level helpers whose names do not contain it
(``_native_conv_vjp``, ``_build_hs_chunked``, ``_build_dinter_chunked``).

CPU-only and import-only: no site is executed, nothing is compiled.
"""

from __future__ import annotations

import ast
import functools
import pathlib

import pytest

from isoexec.ops.compile import bwd_compile

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
ISOEXEC_ROOT = pathlib.Path(bwd_compile.__file__).resolve().parents[2]


# -- AST helpers ---------------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _parse(path: pathlib.Path) -> ast.Module:
    """Cached: the parametrized scope proofs walk every production module once per site."""
    return ast.parse(path.read_text(), filename=str(path))


def _function_nodes(tree: ast.Module) -> dict[str, ast.AST]:
    """``qualname -> node`` for every def in the module, including methods (``Class.method``)."""
    out: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out[f"{node.name}.{sub.name}"] = sub
    return out


def _names_referenced(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _bwd_bindings(tree: ast.Module) -> tuple[set[str], set[str]]:
    """``(bare names, module aliases)`` in this module that resolve to ``bwd_compile.call_region``.

    Resolving the IMPORT rather than matching the spelling matters: the deleted ``autofuse/sites.py``
    defined ``SiteDispatcher.call_region``, an unrelated method of the same name, and matching bare
    ``Name`` nodes alone reported it as an unregistered backward-only site.
    """
    bare: set[str] = set()
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.endswith("bwd_compile"):
                for a in node.names:
                    if a.name == "call_region":
                        bare.add(a.asname or a.name)
            elif mod.endswith("ops.compile"):
                for a in node.names:
                    if a.name == "bwd_compile":
                        modules.add(a.asname or a.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.endswith("compile.bwd_compile"):
                    modules.add(a.asname or a.name.split(".")[-1])
    return bare, modules


def _call_region_sites(node: ast.AST, bindings: tuple[set[str], set[str]]) -> list[str]:
    """Literal site strings passed as ``call_region``'s first argument inside ``node``."""
    bare, modules = bindings
    if not bare and not modules:
        return []
    sites = []
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        fn = call.func
        if isinstance(fn, ast.Name):
            hit = fn.id in bare
        elif isinstance(fn, ast.Attribute):
            hit = fn.attr == "call_region" and isinstance(fn.value, ast.Name) and fn.value.id in modules
        else:
            hit = False
        if not hit or not call.args:
            continue
        first = call.args[0]
        if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
            raise AssertionError(
                f"call_region's site argument must be a string literal so this test can see it; "
                f"got {ast.dump(first)[:80]} at line {first.lineno}"
            )
        sites.append(first.value)
    return sites


def _autograd_backward_qualnames(tree: ast.Module) -> set[str]:
    """``Class.backward`` for every class that subclasses ``torch.autograd.Function``."""
    out = set()
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        bases = {ast.unparse(b) for b in node.bases}
        if not any(b.endswith("autograd.Function") or b == "Function" for b in bases):
            continue
        for sub in node.body:
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name == "backward":
                out.add(f"{node.name}.{sub.name}")
    return out


def _callers_within(funcs: dict[str, ast.AST], target: str) -> set[str]:
    """Functions in this module whose body references ``target``'s bare name."""
    bare = target.rsplit(".", 1)[-1]
    return {q for q, node in funcs.items() if q != target and bare in _names_referenced(node)}


@functools.lru_cache(maxsize=None)
def _production_modules() -> tuple[pathlib.Path, ...]:
    """Every isoexec .py that is not a test and not this file's own package tests."""
    return tuple(
        [p for p in sorted(ISOEXEC_ROOT.rglob("*.py")) if "tests" not in p.parts and not p.name.startswith("test_")]
    )


# -- 1. the table matches the call sites, in both directions --------------------------------------


def test_every_registered_site_is_called_in_its_declared_file():
    """Each ``_BACKWARD_ONLY_SITES`` entry names a file that really calls ``call_region`` with it.

    Catches an entry left behind after its call site moved or was deleted -- the failure mode that
    produced four wrong ledger classifications on 2026-09-04.
    """
    assert bwd_compile._BACKWARD_ONLY_SITES, "the table is empty; the scope proof would be vacuous"
    for site, rel in sorted(bwd_compile._BACKWARD_ONLY_SITES.items()):
        path = REPO_ROOT / rel
        assert path.is_file(), f"site {site!r} declares {rel}, which does not exist"
        tree = _parse(path)
        found = _call_region_sites(tree, _bwd_bindings(tree))
        assert (
            site in found
        ), f"site {site!r} is registered against {rel}, which never calls it (found {sorted(set(found))})"


def test_every_call_region_call_site_in_the_tree_is_registered():
    """The reverse direction: a new ``call_region`` site fails until the table names it.

    This is the half that makes ``bwd_compile.py``'s "adding a site without extending that test
    fails" true. The runtime ``KeyError`` only fires when the flag is on and the line executes;
    this fires at test time on every site, on CPU.
    """
    seen: dict[str, pathlib.Path] = {}
    for path in _production_modules():
        tree = _parse(path)
        for site in _call_region_sites(tree, _bwd_bindings(tree)):
            seen[site] = path
    assert seen, "no call_region call sites found; the walk is broken, not the tree"
    for site, path in sorted(seen.items()):
        assert site in bwd_compile._BACKWARD_ONLY_SITES, (
            f"{path.relative_to(REPO_ROOT)} calls call_region({site!r}) but the site is not in "
            f"_BACKWARD_ONLY_SITES. Register it AND extend this test's scope proof: a site whose "
            f"backward-only reachability has not been asserted must never be compiled."
        )
        declared = REPO_ROOT / bwd_compile._BACKWARD_ONLY_SITES[site]
        assert declared == path, f"site {site!r} is declared against {declared} but called from {path}"


# -- 2. the scope proof itself --------------------------------------------------------------------


@pytest.mark.parametrize("site,rel", sorted(bwd_compile._BACKWARD_ONLY_SITES.items()))
def test_call_site_is_reachable_only_from_an_autograd_backward(site, rel):
    """Walk the static call graph UPWARD from the call site; every root must be a ``backward``.

    This is the whole licence for ``call_region``: a compiled region may reassociate reductions,
    which is legal in a VJP (STATUS: the backward is not under the bitwise contract) and NOT legal
    in a forward. If any root of the caller chain is something other than an
    ``autograd.Function.backward``, some forward can reach the region and the licence is void.
    """
    path = REPO_ROOT / rel
    tree = _parse(path)
    funcs = _function_nodes(tree)
    backwards = _autograd_backward_qualnames(tree)
    assert backwards, f"{rel} registers backward-only sites but defines no autograd.Function.backward"

    bindings = _bwd_bindings(tree)
    hosts = {q for q, node in funcs.items() if site in _call_region_sites(node, bindings)}
    assert hosts, f"no function in {rel} calls call_region({site!r})"

    # Upward closure. A node with no callers inside the module is a ROOT: it must be a backward,
    # or the region is reachable from outside the VJP.
    seen: set[str] = set()
    frontier = set(hosts)
    roots: set[str] = set()
    while frontier:
        current = frontier.pop()
        if current in seen:
            continue
        seen.add(current)
        if current in backwards:
            roots.add(current)  # a backward terminates the walk; do not ascend past it
            continue
        callers = _callers_within(funcs, current)
        if not callers:
            roots.add(current)
            continue
        frontier |= callers - seen

    escapes = roots - backwards
    assert not escapes, (
        f"site {site!r} in {rel} is reachable from {sorted(escapes)}, which is/are not an "
        f"autograd.Function.backward. _BACKWARD_ONLY_SITES claims backward-only scope; a forward "
        f"can reach this region, so its reassociation licence does not hold."
    )


@pytest.mark.parametrize("site,rel", sorted(bwd_compile._BACKWARD_ONLY_SITES.items()))
def test_no_other_production_module_reaches_the_call_site_host(site, rel):
    """The upward walk above is module-local, so close it: no other production module may reference
    the host helper by name. Without this, an import elsewhere would route a forward into the region
    and the module-local proof would not see it."""
    path = REPO_ROOT / rel
    tree = _parse(path)
    funcs = _function_nodes(tree)
    bindings = _bwd_bindings(tree)
    hosts = {q for q, node in funcs.items() if site in _call_region_sites(node, bindings)}
    bare = {h.rsplit(".", 1)[-1] for h in hosts}

    for other in _production_modules():
        if other == path:
            continue
        referenced = _names_referenced(_parse(other)) & bare
        assert not referenced, (
            f"{other.relative_to(REPO_ROOT)} references {sorted(referenced)}, the helper hosting "
            f"call_region({site!r}). The backward-only proof for this site is module-local and this "
            f"reference escapes it."
        )


# -- 3. the runtime guards ------------------------------------------------------------------------


def test_call_region_refuses_an_unregistered_site(monkeypatch):
    """The runtime half of the same rule. Checked before any resolution, so no compile is attempted."""
    monkeypatch.setenv("ISOEXEC_BWD_COMPILE", "1")
    with pytest.raises(KeyError, match="_BACKWARD_ONLY_SITES"):
        bwd_compile.call_region("not.a.registered.site", lambda x: x, 1)


def test_flag_off_is_an_exact_passthrough(monkeypatch):
    """Default (off) must be byte-identical to not having the subsystem: the eager body runs, its
    result is returned unchanged, and nothing is compiled or cached."""
    monkeypatch.setenv("ISOEXEC_BWD_COMPILE", "0")
    assert bwd_compile.bwd_compile_enabled() is False
    sentinel = object()
    calls = []

    def body(*args):
        calls.append(args)
        return sentinel

    site = next(iter(bwd_compile._BACKWARD_ONLY_SITES))
    before = dict(bwd_compile._CACHE)
    got = bwd_compile.call_region(site, body, 1, 2)
    assert got is sentinel and calls == [(1, 2)]
    assert bwd_compile._CACHE == before, "the off path must not populate the artifact cache"


def test_flag_off_passthrough_precedes_the_registration_check(monkeypatch):
    """With the flag off an unregistered site must NOT raise: the guard exists to stop an
    unproven site being COMPILED, and off compiles nothing. A raise here would make the default
    path fail on code the default path does not touch."""
    monkeypatch.setenv("ISOEXEC_BWD_COMPILE", "0")
    assert bwd_compile.call_region("not.a.registered.site", lambda a, b: a + b, 2, 3) == 5


def test_registered_files_are_inside_isoexec():
    """A table entry pointing outside the package would put the proof beyond this test's walk."""
    for site, rel in sorted(bwd_compile._BACKWARD_ONLY_SITES.items()):
        resolved = (REPO_ROOT / rel).resolve()
        assert ISOEXEC_ROOT in resolved.parents, f"site {site!r} declares {rel}, outside {ISOEXEC_ROOT}"
