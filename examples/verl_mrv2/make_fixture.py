"""Build the fixed four-prompt GSM8K fixture for the verl x IsoExec x vLLM Model Runner V2 cases.

The rows use verl's own GSM8K preprocessing (examples/data_preprocess/gsm8k.py): the question plus
its step-by-step instruction as one user turn, rule reward with the extracted ground truth. The
script tokenizes each chat-templated prompt with the exact model tokenizer and refuses a fixture
whose prompts would be truncated at 512 tokens, are not all distinct in length, or would be
replicated instead of split across two PCP ranks (vLLM 0.33 ``PCPManager.replicated_requests``
with DCP=2). It writes ``gsm8k-4.parquet`` and ``fixture.json`` (row indices, token lengths, hashes).

Usage:
    python examples/verl_mrv2/make_fixture.py --out DIR --tokenizer MODEL_DIR [--model MODEL_DIR ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'
DATASET, CONFIG, SPLIT = "openai/gsm8k", "main", "train"
ROWS = (0, 1, 2, 3)
MAX_PROMPT_LENGTH = 512
PCP, DCP = 2, 2


def extract_solution(answer: str) -> str:
    match = re.search("#### (\\-?[0-9\\.\\,]+)", answer)
    if match is None:
        raise ValueError(f"GSM8K answer without a final '####' value: {answer[-80:]!r}")
    return match.group(0).split("#### ")[1].replace(",", "")


def pcp_split(query_len: int, pcp: int = PCP, dcp: int = DCP) -> bool:
    """True when vLLM splits this prefill across PCP ranks instead of replicating it."""
    chunks = 2 * pcp
    chunk = (query_len + chunks - 1) // chunks
    return not (dcp > 1 and (chunks - 1) * chunk >= query_len)


def rollout_prompt_ids(model_path: str, conversations: list[list[dict]]) -> list[list[int]]:
    """Prompt token IDs exactly as verl's agent loop builds them (Continuous Token builder)."""
    from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder
    from verl.workers.config import HFModelConfig

    model = HFModelConfig(path=model_path)
    model_type = getattr(model.hf_config, "model_type", None)
    builder = create_continuous_token_builder(model.tokenizer, hf_model_type=model_type if isinstance(model_type, str)
                                              else None, chat_template_kwargs={}, mm_processor_kwargs={},
                                              processor=model.processor)
    return [list(builder.build_initial_tokens(messages)) for messages in conversations]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tokenizer", action="append", required=True,
                        help="local model directory whose tokenizer scores the prompts (repeatable)")
    parser.add_argument("--dataset-path", default=None, help="local GSM8K dataset path for datasets.load_dataset")
    args = parser.parse_args()

    import datasets
    import pandas as pd

    dataset = datasets.load_dataset(args.dataset_path or DATASET, CONFIG, split=SPLIT)
    records = []
    for index in ROWS:
        example = dataset[index]
        records.append({
            "data_source": DATASET,
            "prompt": [{"role": "user", "content": example["question"] + " " + INSTRUCTION}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": extract_solution(example["answer"])},
            "extra_info": {"split": SPLIT, "index": index, "answer": example["answer"], "question": example["question"]},
        })
    tokenizers = {}
    for path in args.tokenizer:
        lengths = [len(ids) for ids in rollout_prompt_ids(path, [record["prompt"] for record in records])]
        if max(lengths) > MAX_PROMPT_LENGTH:
            raise SystemExit(f"{path}: a prompt has {max(lengths)} tokens > {MAX_PROMPT_LENGTH}")
        if len(set(lengths)) != len(lengths):
            raise SystemExit(f"{path}: prompt lengths {lengths} are not distinct")
        if not all(pcp_split(length) for length in lengths):
            raise SystemExit(f"{path}: some prompts would be replicated across PCP ranks: {lengths}")
        tokenizers[path] = {"prompt_tokens": lengths, "pcp_split": [pcp_split(n) for n in lengths]}
    args.out.mkdir(parents=True, exist_ok=True)
    parquet = args.out / "gsm8k-4.parquet"
    pd.DataFrame.from_records(records).to_parquet(parquet, index=False)
    receipt = {
        "dataset": DATASET, "config": CONFIG, "split": SPLIT, "rows": list(ROWS),
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "parquet": parquet.name, "parquet_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
        "content_sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
        "max_prompt_length": MAX_PROMPT_LENGTH, "pcp": PCP, "dcp": DCP, "tokenizers": tokenizers,
    }
    (args.out / "fixture.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
