#!/usr/bin/env python3
"""Train Qwen3.5-0.8B for one step through SkyRL's Tinker API on IsoExec, in the runtime `make setup` created.

Needs one SM90 GPU, or two with --no-colocate. Starts SkyRL's Tinker API server on IsoExec's backend with the
full comparison on, then acts as the Tinker client: synchronizes the weights, samples each prompt at
temperature 1, runs an importance_sampling forward_backward on the sampled sequences and an optim_step,
synchronizes the new weights, samples again and runs a second forward_backward. Every loss-active response row
is compared in a policy forward before each backward, so exit code 0 is the result. The advantages are all 1:
this checks the path, not learning. --no-isoexec runs the same client on SkyRL's own backend. Extra key=value
arguments go to the server's backend_config, for example generator.inference_engine.gpu_memory_utilization=0.3.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from run_tiny import download_model, runtime

# A long context shared by both prompts: its prefill chunks cross vLLM's mamba block boundaries, where CPR keeps
# the checkpoints a prefix-cache hit resumes from, and the second prompt and every repeat can hit the first.
CONTEXT = " ".join(f"Fact {i}: {i} is {'even' if i % 2 == 0 else 'odd'}." for i in range(80))
PROMPTS = [
    CONTEXT + "\n\nNatalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
    "How many clips did Natalia sell altogether in April and May? Explain each step, then give the answer.",
    CONTEXT + "\n\nWeng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. "
    "How much did she earn? Explain each step, then give the answer.",
]
SAMPLES_PER_PROMPT = 4
MAX_TOKENS = 48
LEARNING_RATE = 1e-6


def backend_config(isoexec: bool, colocate: bool) -> dict:
    """BF16, Megatron TP1/PP1 and one vLLM engine (Model Runner V2, FULL_AND_PIECEWISE, prefix caching and
    chunked prefill), colocated on one GPU or on its own second GPU."""
    config = {
        'trainer.bf16': True, 'trainer.policy.language_model_only': True,
        'trainer.gradient_checkpointing': False, 'trainer.remove_microbatch_padding': True,
        'trainer.fused_lm_head_logprob': False, 'trainer.mtp.enabled': False,
        'trainer.placement.colocate_all': colocate, 'trainer.placement.policy_num_nodes': 1,
        'trainer.placement.policy_num_gpus_per_node': 1,
        'trainer.policy.megatron_config.tensor_model_parallel_size': 1,
        'trainer.policy.megatron_config.pipeline_model_parallel_size': 1,
        'trainer.policy.megatron_config.context_parallel_size': 1,
        'trainer.policy.megatron_config.expert_model_parallel_size': 1,
        'trainer.policy.megatron_config.expert_tensor_parallel_size': 1,
        'trainer.policy.megatron_config.transformer_config_kwargs.sequence_parallel': False,
        'trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload': False,
        'trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer': False,
        'trainer.micro_train_batch_size_per_gpu': 1, 'trainer.micro_forward_batch_size_per_gpu': 1,
        'trainer.algorithm.temperature': 1.0, 'trainer.algorithm.max_seq_len': 2048,
        'generator.inference_engine.backend': 'vllm', 'generator.inference_engine.language_model_only': True,
        'generator.inference_engine.run_engines_locally': True, 'generator.inference_engine.num_engines': 1,
        'generator.inference_engine.tensor_parallel_size': 1, 'generator.inference_engine.pipeline_parallel_size': 1,
        'generator.inference_engine.data_parallel_size': 1, 'generator.inference_engine.expert_parallel_size': 1,
        'generator.inference_engine.weight_sync_backend': 'nccl',
        'generator.inference_engine.gpu_memory_utilization': 0.35,
        'generator.inference_engine.enforce_eager': False,
        'generator.inference_engine.enable_prefix_caching': True,
        # 32 tokens per step splits every prompt, so each prefill is chunked.
        'generator.inference_engine.max_num_batched_tokens': 32,
        'generator.inference_engine.engine_init_kwargs': {
            'max_model_len': 2048, 'max_num_seqs': 8, 'enable_chunked_prefill': True,
            'compilation_config': {'mode': 0, 'cudagraph_mode': 'FULL_AND_PIECEWISE'},
        },
    }
    if colocate:
        config['trainer.policy.optimizer_config.offload_after_step'] = True
    if isoexec:
        config['isoexec'] = {'model_package': 'isoexec.models.qwen3_5', 'full_logprob_comparison': True}
        # As run_tiny.py: IsoExec's attention backward is deterministic under it. SkyRL's own Qwen3.5 model
        # refuses it unless its GDN kernel backend is 'torch', so the run without IsoExec leaves it off.
        config['trainer.policy.megatron_config.transformer_config_kwargs.deterministic_mode'] = True
    return config


def wait_for_server(base_url: str, server: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            sys.exit(f'[tinker-tiny] the Tinker API server exited with {server.returncode} before it was healthy')
        try:
            with urlopen(f'{base_url}/api/v1/healthz', timeout=5) as response:
                if json.loads(response.read()).get('status') == 'ok':
                    return
        except (URLError, OSError, ValueError):
            pass
        time.sleep(5)
    sys.exit(f'[tinker-tiny] the Tinker API server was not healthy after {timeout:.0f}s')


def encode_prompt(tokenizer, text: str) -> list[int]:
    tokens = tokenizer.apply_chat_template([{'role': 'user', 'content': text}], tokenize=True,
                                           add_generation_prompt=True, return_dict=False)
    return list(tokens['input_ids'] if isinstance(tokens, dict) else tokens)


def datum(types, prompt: list[int], response: list[int], logprobs: list[float]):
    """The sampled sequence as next-token training data: prompt targets carry no loss."""
    tokens = prompt + response
    pad = [0.0] * (len(prompt) - 1)

    def tensor(values, dtype):
        return types.TensorData(data=values, dtype=dtype, shape=[len(values)])

    return types.Datum(model_input=types.ModelInput.from_ints(tokens[:-1]), loss_fn_inputs={
        'target_tokens': tensor(tokens[1:], 'int64'),
        'weights': tensor(pad + [1.0] * len(response), 'float32'),
        'logprobs': tensor(pad + list(logprobs), 'float32'),
        'advantages': tensor(pad + [1.0] * len(response), 'float32'),
    })


def train_round(types, training_client, prompts: list[list[int]], round_index: int, isoexec: bool) -> None:
    """Synchronize, sample every prompt, and run one compared forward_backward on the samples."""
    sampling_client = training_client.save_weights_and_get_sampling_client()
    futures = [(prompt, sampling_client.sample(
        prompt=types.ModelInput.from_ints(prompt), num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=MAX_TOKENS, temperature=1.0, top_p=1.0, top_k=-1,
                                             seed=round_index * 1000 + index),
    )) for prompt in prompts for index in range(SAMPLES_PER_PROMPT)]
    data = []
    for prompt, future in futures:
        sequence = future.result().sequences[0]
        if sequence.tokens:
            data.append(datum(types, prompt, list(sequence.tokens), list(sequence.logprobs)))
    if not data:
        sys.exit('[tinker-tiny] every sample was empty')
    # Tinker's policy-gradient loss. Not 'ppo': SkyRL's backend maps it to ppo_policy_loss, which asserts that the
    # config's policy_loss_type is 'regular' or 'dual_clip', and SkyRL defaults policy_loss_type to 'dppo'.
    metrics = training_client.forward_backward(data, 'importance_sampling').result().metrics
    print(f'[tinker-tiny] round {round_index}: {len(data)} sequences, forward_backward metrics {metrics}', flush=True)
    if isoexec and not metrics.get('isoexec/full_logprobs_verified_rows:sum'):
        sys.exit('[tinker-tiny] forward_backward reported no compared rows')


def post(base_url: str, path: str, payload: dict, timeout: float) -> dict:
    request = Request(f'{base_url}{path}', data=json.dumps(payload).encode(), method='POST',
                      headers={'accept': 'application/json', 'content-type': 'application/json'})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def unload(base_url: str, model_id: str) -> None:
    """Delete the model on the server, which also stops the Ray cluster its engine started."""
    future = post(base_url, '/api/v1/unload_model', {'model_id': model_id}, timeout=30)
    while True:
        try:
            post(base_url, '/api/v1/retrieve_future', {'request_id': future['request_id']}, timeout=600)
            return
        except HTTPError as error:
            if error.code != 408:  # 408: not done within the server's wait; ask again
                raise


def run_client(base_url: str, model: Path, isoexec: bool) -> None:
    import tinker
    from tinker import types

    service_client = tinker.ServiceClient(base_url=base_url, api_key='tml-dummy')
    training_client = None
    try:
        # rank 0 selects full-parameter training on SkyRL's backend.
        training_client = service_client.create_lora_training_client(base_model=str(model), rank=0)
        tokenizer = training_client.get_tokenizer()
        prompts = [encode_prompt(tokenizer, text) for text in PROMPTS]
        train_round(types, training_client, prompts, 0, isoexec)
        result = training_client.optim_step(types.AdamParams(learning_rate=LEARNING_RATE)).result()
        print(f'[tinker-tiny] optim_step metrics {result.metrics}', flush=True)
        train_round(types, training_client, prompts, 1, isoexec)
    finally:
        if training_client is not None:
            unload(base_url, training_client.model_id)
        service_client.holder.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('ISOEXEC_ROOT', '~/.isoexec')))
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--isoexec', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--colocate', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--startup-timeout', type=float, default=1800.0)
    args, extra = parser.parse_known_args()
    root = args.root.expanduser().resolve()
    env, python, skyrl = runtime(root)
    model = download_model(python, env)
    config = backend_config(args.isoexec, args.colocate)
    for item in extra:
        key, _, value = item.partition('=')
        config[key] = json.loads(value)
    # Each run keeps its own request database and sampler checkpoints (SkyRL defaults to one shared file and /tmp).
    run_dir = root / 'runs' / f"tinker-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True)
    # The Tinker API restarts its engine with `uv run`; from this preinstalled runtime it must be started the
    # same way, and the engine then runs on this interpreter (SkyRL's _build_uv_run_cmd_engine).
    command = ['uv', 'run', '--isolated', '--no-project', '--', str(python), '-m', 'skyrl.tinker.api',
               '--base-model', str(model), '--backend', 'megatron', '--runtime-role', 'combined',
               '--port', str(args.port), '--database-url', f'sqlite:///{run_dir / "tinker.db"}',
               '--checkpoints-base', str(run_dir / 'checkpoints'),
               '--backend-config', json.dumps(config, separators=(',', ':'))]
    print('[tinker-tiny] ' + ' '.join(command), flush=True)
    server = subprocess.Popen(command, cwd=skyrl, env=env, start_new_session=True)
    base_url = f'http://127.0.0.1:{args.port}'
    try:
        wait_for_server(base_url, server, args.startup_timeout)
        run_client(base_url, model, args.isoexec)
    finally:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
    print(f'[tinker-tiny] PASS isoexec={args.isoexec} colocate={args.colocate}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
