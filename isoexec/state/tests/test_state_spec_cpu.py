from dataclasses import replace

import pytest

from isoexec.compiler.contract.contract import ModelContract
from isoexec.compiler.contract.validation import validate_model_contract
from isoexec.schema import ShardAxis, col, replicated
from isoexec.state.contract import StateDecl, StateRef, StateSpec
from isoexec.state.tests._fixtures import REGISTERED_OP_IDS, single_spec_contract
from isoexec.state.validation import ContractError, validate_state_spec

TOKEN = StateSpec("history", (4, 32), "bf16", col(dim=0), "per_token", 1, "full_history")
REQUEST = StateSpec("value", (4, 16, 32), "fp32", replicated(), "per_request")


@pytest.mark.parametrize(
    "spec",
    [
        TOKEN,
        replace(TOKEN, tokens_per_entry=2, retention="sliding_window", window_tokens=8),
        REQUEST,
        replace(REQUEST, shard=col(dim=2)),
    ],
)
def test_valid_state_spec(spec):
    validate_state_spec(spec)


@pytest.mark.parametrize(
    "field,value",
    [
        ("tokens_per_entry", 1),
        ("retention", "full_history"),
        ("window_tokens", 8),
    ],
)
def test_per_request_rejects_history_fields(field, value):
    with pytest.raises(ContractError, match=f"per_request forbids {field}"):
        validate_state_spec(replace(REQUEST, **{field: value}))


@pytest.mark.parametrize("value", [None, 0, -1, True, 1.0, "1"])
def test_per_token_requires_positive_integer_tokens_per_entry(value):
    with pytest.raises(ContractError, match="tokens_per_entry"):
        validate_state_spec(replace(TOKEN, tokens_per_entry=value))


@pytest.mark.parametrize("value", [None, "all", "align", "none"])
def test_per_token_requires_known_retention(value):
    with pytest.raises(ContractError, match="retention"):
        validate_state_spec(replace(TOKEN, retention=value))


@pytest.mark.parametrize("value", [0, 8])
def test_full_history_forbids_window(value):
    with pytest.raises(ContractError, match="full_history forbids window_tokens"):
        validate_state_spec(replace(TOKEN, window_tokens=value))


@pytest.mark.parametrize("value", [None, 0, -1, True, 8.0, "8"])
def test_sliding_window_requires_positive_integer_window(value):
    with pytest.raises(ContractError, match="window_tokens"):
        validate_state_spec(replace(TOKEN, retention="sliding_window", window_tokens=value))


def test_partial_retention_refuses_without_rounding():
    with pytest.raises(ContractError, match="partial retention"):
        validate_state_spec(replace(TOKEN, tokens_per_entry=2, retention="sliding_window", window_tokens=7))


@pytest.mark.parametrize("shape", [(), (0,), (-1, 32), (True, 32), (4.0, 32), ("4", 32), [4, 32], None])
def test_shape_requires_immutable_positive_concrete_dimensions(shape):
    with pytest.raises(ContractError, match="shape"):
        validate_state_spec(replace(TOKEN, shape=shape))


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", ""),
        ("name", " "),
        ("name", 42),
        ("dtype", ""),
        ("dtype", None),
        ("granularity", "per_token_paged"),
        ("granularity", "per_request_slot"),
        ("granularity", None),
    ],
)
def test_invalid_scalar_fields_refuse(field, value):
    with pytest.raises(ContractError, match=field):
        validate_state_spec(replace(TOKEN, **{field: value}))


@pytest.mark.parametrize(
    "shard",
    [
        ShardAxis("row", dim=1),
        ShardAxis("expert"),
        ShardAxis("unknown"),
        col(dim=-1),
        col(dim=2),
        col(dim=None),
        col(dim=True),
        col(dim=0.0),
        col(stride=2),
        col(stride=True),
        col(stride=1.0),
        col(units=4),
        col(segments=((4, None),)),
        replace(col(), reduce="sum_over_tp"),
        replicated(reduce="sum_over_tp"),
        ShardAxis("replicated", dim=0),
        ShardAxis("replicated", units=1),
        ShardAxis("replicated", stride=2),
        ShardAxis("replicated", segments=()),
        None,
    ],
)
def test_unsupported_state_sharding_refuses(shard):
    with pytest.raises(ContractError):
        validate_state_spec(replace(TOKEN, shard=shard))


@pytest.mark.parametrize(
    "contract,message",
    [
        (
            ModelContract((StateDecl("state", (TOKEN,)),) * 2, (StateRef("block", "gdn.core", "state"),)),
            "duplicate StateDecl",
        ),
        (
            ModelContract((StateDecl("state", (TOKEN, TOKEN)),), (StateRef("block", "gdn.core", "state"),)),
            "duplicate StateSpec",
        ),
        (ModelContract((StateDecl("state", ()),), (StateRef("block", "gdn.core", "state"),)), "non-empty specs"),
        (ModelContract((StateDecl("state", [TOKEN]),), (StateRef("block", "gdn.core", "state"),)), "specs tuple"),
        (ModelContract((StateDecl("state", (TOKEN,)),), (StateRef("block", "gdn.core", "missing"),)), "unresolved"),
        (
            ModelContract((StateDecl("state", (TOKEN,)),), (StateRef("block", "unknown", "state"),)),
            "unknown StateRef.op_id",
        ),
        (
            ModelContract((StateDecl("state", (TOKEN,)),), (StateRef("block", "gdn.core", "state"),) * 2),
            "duplicate StateRef",
        ),
        (
            ModelContract(
                (StateDecl("first", (TOKEN,)), StateDecl("second", (TOKEN,))),
                (StateRef("block", "attention.core.gqa", "first"), StateRef("block", "attention.core.gqa", "second")),
            ),
            "duplicate StateRef",
        ),
        (
            ModelContract(
                (StateDecl("state", (TOKEN,)), StateDecl("unused", (REQUEST,))),
                (StateRef("block", "attention.core.gqa", "state"),),
            ),
            "no StateRef",
        ),
        (ModelContract((None,), ()), "expected a StateDecl"),
        (ModelContract((), (None,)), "expected a StateRef"),
        (ModelContract((StateDecl("state", (None,)),), ()), "expected a StateSpec"),
        (ModelContract((StateDecl("state", (TOKEN,)),), (StateRef(None, "gdn.core", "state"),)), "module_path"),
        (ModelContract((StateDecl("state", (TOKEN,)),), ()), "no StateRef"),
        (ModelContract([], ()), "must be tuples"),
        (ModelContract((), []), "must be tuples"),
    ],
)
def test_invalid_declarations_and_references_refuse(contract, message):
    with pytest.raises(ContractError, match=message):
        validate_model_contract(contract, op_ids=REGISTERED_OP_IDS)


def test_multiple_refs_to_one_state_are_unsupported():
    contract = single_spec_contract(TOKEN)
    ref = replace(contract.state_refs[0], module_path="other")
    with pytest.raises(ContractError, match="multiple StateRefs"):
        validate_model_contract(replace(contract, state_refs=contract.state_refs + (ref,)), op_ids=REGISTERED_OP_IDS)


def test_contract_tuple_order_is_not_a_reference_key():
    contract = ModelContract(
        (StateDecl("first", (TOKEN,)), StateDecl("second", (REQUEST,))),
        (StateRef("a", "attention.core.gqa", "first"), StateRef("b", "gdn.core", "second")),
    )
    reordered = replace(contract, state_decls=contract.state_decls[::-1], state_refs=contract.state_refs[::-1])

    validate_model_contract(reordered, op_ids=REGISTERED_OP_IDS)


def test_empty_model_contract_is_valid():
    validate_model_contract(ModelContract((), ()), op_ids=REGISTERED_OP_IDS)


def test_segment_scope_is_a_per_token_sliding_window():
    window = replace(TOKEN, retention="sliding_window", window_tokens=8, scope="segment")
    validate_state_spec(window)
    assert TOKEN.scope == REQUEST.scope == "request"                 # the default: every existing declaration
    for bad in (replace(TOKEN, scope="segment"), replace(REQUEST, scope="segment"), replace(TOKEN, scope="turn")):
        with pytest.raises(ContractError):
            validate_state_spec(bad)
