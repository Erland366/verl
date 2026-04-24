---
name: verl-rocm-mi210-native-vllm
description: >
  Native ROCm debugging and launch guidance for verl on MI210/gfx90a with
  local vLLM source checkout and 4-GPU async rollout.
metadata:
  short-description: "verl + ROCm vLLM on MI210"
---

# Skill: verl-rocm-mi210-native-vllm

## When to use

Use this when:
- Running `verl` natively on AMD MI210 / `gfx90a`
- Using conda env `verlrl`
- Debugging 4-GPU PPO with ROCm `vLLM`
- Seeing startup failures in Ray, Triton, or external-DP `vLLM`

## Validated context

- Host GPU: AMD Instinct MI210
- Arch: `gfx90a`
- Env: `conda activate verlrl`
- PyTorch: `2.7.1+rocm6.3`
- Ray: `2.54.1`
- verl: `0.8.0.dev`
- vLLM path: local source checkout under `/tmp/vllm-v0.10.0`
- AMD SMI Python path: `/opt/rocm-6.3.3/share/amd_smi`
- W&B path used in validation: logged in and writing to entity `erlandpg`

## Recommended launch shape

Use:
- `algorithm.adv_estimator=grpo`
- `actor_rollout_ref.rollout.tensor_model_parallel_size=1`
- `actor_rollout_ref.rollout.data_parallel_size=4`
- `+actor_rollout_ref.rollout.engine_kwargs.vllm.data_parallel_external_lb=True`

Prefer:
- `ATTN_IMPLEMENTATION=eager`
- `+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.mode=0`
- `+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.backend=eager`
- `+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=NONE`
- `actor_rollout_ref.rollout.free_cache_engine=False`
- `+actor_rollout_ref.rollout.enable_sleep_mode=False`
- `actor_rollout_ref.actor.use_torch_compile=False`
- `actor_rollout_ref.actor.fsdp_config.use_torch_compile=False`
- `actor_rollout_ref.ref.use_torch_compile=False`
- `actor_rollout_ref.ref.fsdp_config.use_torch_compile=False`
- `critic.fsdp.use_torch_compile=False`
- `actor_rollout_ref.model.use_remove_padding=True`
- `critic.model.use_remove_padding=True`

Set or preserve:
- `RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1`
- `HSA_FORCE_FINE_GRAIN_PCIE=1`
- `RCCL_MSCCL_ENABLE=0`
- `RCCL_MSCCLPP_ENABLE=0`

## Known-good progress markers

Healthy startup should reach:
- Ray local instance starts successfully
- PPO config validation passes
- train/val prompt filtering completes
- FSDP actor/ref/critic initialization completes
- external-DP `vLLM` server launch begins
- `vLLM` logs `enable_sleep_mode: False`
- `vLLM` V1 engine initializes
- W&B run starts and `AgentLoopWorker` actors start
- first external-DP weight update logs `update_weights done`

## What worked

- Core native `verlrl` installation is usable on this MI210 host
- W&B login and `erlandpg` logging setup are not blockers
- Source-checkout `vLLM` version discovery works via `vllm.__version__`
- `verl` fallback to visible accelerator id is sufficient when ROCm `vLLM` lacks device UUID lookup
- Fresh source build of `vLLM` 0.10.x works with PyTorch 2.7.1 + ROCm 6.3 on MI210
- Native 4-GPU smoke launcher works best with `TP=1`, `DP=4`, external DP load balancing, eager execution settings, sleep mode disabled, and torch compile disabled
- External-DP weight transfer can progress when each TP group leader drives the matching server-rank collective receiver

## How this was made to work

The successful path came from treating the problem as a sequence of startup gates instead of a single "ROCm is broken" failure.

1. Prove the base environment first
- Activate `verlrl` directly; do not use `uv run`.
- Confirm PyTorch is the ROCm build (`2.7.1+rocm6.3` in the validated run).
- Confirm the host exposes MI210 / `gfx90a` devices with `rocm-smi`.
- Confirm Ray can start locally before debugging vLLM payloads.

2. Replace the stale vLLM path with a matching ROCm source build
- The stale `/tmp/vllm-v0.12.0` path was not the right base for this environment.
- Use `/tmp/vllm-v0.10.0` built for ROCm/gfx90a.
- Validate imports before running `verl`:
  - `import vllm`
  - `from vllm import _C, _rocm_C`
  - `from vllm.engine.arg_utils import AsyncEngineArgs`
  - `from vllm.v1.engine.async_llm import AsyncLLM`

3. Make source-checkout vLLM detectable
- Editable/source vLLM may not expose normal package metadata.
- When `importlib.metadata.version("vllm")` fails, fall back to importing `vllm.__version__`.

4. Normalize accelerator visibility for vLLM child processes
- ROCm still uses CUDA-shaped API and env names in parts of PyTorch/vLLM.
- Ray/vLLM child processes can fail if both `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` are present and disagree.
- Before spawning each vLLM server, set one visible-device env var and remove conflicting inherited variables.

5. Adapt external-DP launch to vLLM 0.10 semantics
- vLLM 0.10 does not expose the newer `--data-parallel-external-lb` CLI flag.
- For vLLM `<0.12`, remove `data_parallel_external_lb` from CLI args and rely on explicit `--data-parallel-rank`.
- Propagate `data_parallel_master_port` from rank 0 to every external-DP server before creating the vLLM engine config.
- Launch one local vLLM server per DP rank for the 4x MI210 smoke path (`TP=1`, `DP=4`).

6. Fix the external-DP collective ownership
- The key hang was weight transfer, not model loading.
- With external DP, each server rank needs the matching rollout rank to drive `collective_rpc`.
- Map `server_rank = rollout_rank // tensor_model_parallel_size`.
- Only TP group leaders drive collectives: `rollout_rank % tensor_model_parallel_size == 0`.
- For the validated `TP=1`, this means every rollout rank drives the matching vLLM server receiver.

7. Disable ROCm-unsupported runtime features
- vLLM sleep mode is unsupported on this ROCm platform.
- Use `actor_rollout_ref.rollout.free_cache_engine=False`.
- Use `+actor_rollout_ref.rollout.enable_sleep_mode=False`.
- Disable actor/ref/critic torch compile.
- Use vLLM eager settings:
  - `compilation_config.mode=0`
  - `compilation_config.backend=eager`
  - `compilation_config.cudagraph_mode=NONE`

8. Keep remove padding enabled, but remove the CUDA `flash_attn` dependency
- Setting `USE_REMOVE_PADDING=False` reproduced early silent worker death during FSDP worker initialization.
- Keeping remove padding enabled reached old-log-prob computation, then failed because native ROCm lacked `flash_attn.bert_padding`.
- Add a fallback in `verl/utils/attention_utils.py` to use Hugging Face padding helpers and `einops` when `flash_attn.bert_padding` is unavailable.

9. Use GRPO as the smoke target
- `ADV_ESTIMATOR=gae` brings up the critic path and still reproduces silent Ray worker death.
- `ADV_ESTIMATOR=grpo` avoids the separate critic and gives a working end-to-end rollout/training baseline.
- Treat PPO/GAE critic startup as a separate debugging target, not as the first smoke-test gate.

10. Preserve observability and cleanup
- Run a 5-minute foreground validation before full tmux execution.
- Run the full command in tmux and keep the pane history intact.
- Let W&B log the run with `trainer.logger='["console","wandb"]'`.
- After completion, check `ray status`, `rocm-smi --showpids --showmemuse`, and stale multiprocessing workers. Kill only confirmed stale vLLM worker parent/child processes.

## Validated smoke run: 2026-04-24

The native MI210 GRPO smoke path completed successfully on 4x MI210.

- Command: `scripts/run_qwen25_3b_4gpu_smoke.sh`
- Env: `conda activate verlrl`
- vLLM source: `/tmp/vllm-v0.10.0`
- Model: `Qwen/Qwen2.5-3B-Instruct`
- Dataset: GSM8K parquet
- Train samples: `32`
- Batch size: `8`
- Total training steps: `4`
- Rollout: async vLLM
- TP: `1`
- DP: `4`
- External DP LB: enabled
- Algorithm: `grpo`
- Result: completed `4/4` steps in `7m42s`

Observed step timings:

| Step | Step Time | Throughput | Notes |
|------|----------:|-----------:|-------|
| 1 | `281.04s` | `1.64` | Includes first-step rollout/server warmup cost |
| 2 | `60.09s` | `7.62` | Stable steady-state step |
| 3 | `61.08s` | `7.48` | Stable steady-state step |
| 4 | `60.11s` | `7.22` | Stable steady-state step |

The warning about Qwen2.5 sliding-window attention on ROCm Triton flash attention was non-fatal in this smoke run.

## Post-run cleanup

After successful completion, verify that vLLM subprocesses did not survive Ray shutdown:

```bash
rocm-smi --showpids --showmemuse --showuse --showmeminfo vram
```

If Python multiprocessing children still hold VRAM after the run has returned to the shell and `ray status` reports no active Ray instance, inspect them:

```bash
ps -o pid,ppid,stat,etime,cmd -p <pid>
ps --ppid <parent_pid> -o pid,ppid,stat,etime,cmd
```

Kill only confirmed stale vLLM worker parent/child processes. Re-check `rocm-smi` and confirm all MI210s return to `VRAM%: 0`.

## Failure ladder

1. `hf` async rollout not found
- Current PPO path requires async rollout backends.
- Use `vllm`, not `hf`, for end-to-end PPO on this stack.

2. Triton ABI mismatches on ROCm PyTorch 2.7
- Observed missing pieces included:
  - `triton_key`
  - `CompiledKernel.launch_enter_hook`
  - `CompiledKernel.launch_exit_hook`
  - `cluster_dims`
- Local compatibility shims were required in the active environment.

3. Mixed HIP/CUDA visibility envs
- Symptom: Ray AMD import failure complaining about inconsistent `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES`.
- Fix: normalize each `vLLM` server actor to a single visible-device env var before engine subprocess spawn.

4. External-DP rank startup errors
- External-DP mode required explicit `data_parallel_rank`.
- Remote external-DP engines must not be launched with `--headless`.
- `vLLM` 0.10 infers external-DP load balancing from `--data-parallel-rank` and does not expose the newer `--data-parallel-external-lb` CLI flag.

5. External-DP handshake metadata
- Symptom:
  - `msgspec.ValidationError: Expected int | str | array, got None - at $.parallel_config[...]`
- Fix:
  - propagate `data_parallel_master_port` from rank 0 to every external-DP server before creating the vLLM engine config.

6. ROCm sleep mode
- Symptom:
  - `Value error, Sleep mode is not supported on current platform.`
- Fix:
  - set `actor_rollout_ref.rollout.free_cache_engine=False`
  - set `+actor_rollout_ref.rollout.enable_sleep_mode=False`

7. Native smoke path remove-padding dependency
- Symptom:
  - `ModuleNotFoundError: No module named 'flash_attn'` from `verl/utils/attention_utils.py` during old-log-prob computation
- Cause:
  - `actor_rollout_ref.model.use_remove_padding=True` or `critic.model.use_remove_padding=True` takes the CUDA `flash_attn.bert_padding` path; the native `verlrl` ROCm environment does not provide that module.
- Fix:
  - keep `USE_REMOVE_PADDING=True` for the MI210 native smoke launcher, and rely on the explicit `verl/utils/attention_utils.py` fallback to Hugging Face padding helpers when `flash_attn.bert_padding` is unavailable.
  - avoid setting `USE_REMOVE_PADDING=False` on this smoke path unless debugging a separate model issue; it reproduced early silent worker death during FSDP worker initialization.

8. PPO/GAE critic startup on the native MI210 smoke stack
- Symptom:
  - `ray.exceptions.ActorDiedError` during `trainer.init_workers()` with one `WorkerDict` process disappearing silently and peer ranks reporting TCPStore connection resets.
- Current status:
  - Reproduced with `algorithm.adv_estimator=gae` while bringing up the colocated actor/ref/critic stack.
- Workaround:
  - default the native GSM8K smoke launcher to `ADV_ESTIMATOR=grpo`, which avoids the separate critic model.
  - keep `ADV_ESTIMATOR=gae` as an explicit opt-in only for focused critic debugging.

## Recommended practice

- Always do a foreground validation run before a tmux overnight run.
- Treat post-startup handshake failures as integration bugs, not installation bugs.
- Stop broad environment churn once startup passes the earlier gates; debug the exact vLLM/verl payload next.
- Use `ADV_ESTIMATOR=grpo` as the default native MI210 smoke path until the PPO/GAE critic startup failure is fixed.

## When to abandon this path

Switch frameworks if:
- You need usable RL training soon
- You are not prepared to patch both `verl` and local ROCm `vLLM`
- Targeted startup fixes still do not produce a first PPO training step

## Next debugging target

If extending beyond the validated smoke run:
- keep `tmux` logs intact
- run a longer GRPO job to check steady-state stability
- debug `ADV_ESTIMATOR=gae` separately because critic initialization still reproduces silent Ray worker death
- consider `VLLM_USE_TRITON_FLASH_ATTN=0` only if Qwen2.5 ROCm sliding-window attention warnings become runtime failures
- avoid reopening already-fixed issues such as env visibility conflicts, headless external-DP launch, missing `data_parallel_master_port`, ROCm sleep mode, external-DP weight transfer, missing CUDA `flash_attn`, or the known PPO/GAE critic startup crash
