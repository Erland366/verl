# Experiment Log

## 2026-04-24 - Native MI210 GRPO vLLM Smoke Run

General description: validated a native ROCm `verl` smoke path on 4x AMD MI210 using the existing `verlrl` conda environment and a local vLLM 0.10 source checkout.

Setup:
- Env: `conda activate verlrl`
- PyTorch: `2.7.1+rocm6.3`
- Ray: `2.54.1`
- vLLM source: `/tmp/vllm-v0.10.0`
- Model: `Qwen/Qwen2.5-3B-Instruct`
- Dataset: GSM8K parquet
- Algorithm: `grpo`
- Rollout: async `vllm`
- TP/DP: `1/4`
- External DP LB: enabled
- Sleep/free-cache: disabled
- Torch compile: disabled
- Remove padding: enabled with Hugging Face padding fallback when `flash_attn.bert_padding` is unavailable

Result:
- Completed `4/4` training steps in `tmux-7`.
- Total progress line: `Training Progress: 100%|...| 4/4 [07:42<00:00, 115.66s/it]`.
- W&B local run path: `wandb/run-20260424_110356-8lfzrqlr`.
- W&B run id: `8lfzrqlr`.
- Final GPUs returned to `VRAM%: 0` after stale vLLM worker cleanup.

Key lessons:
- Use `ADV_ESTIMATOR=grpo` as the default MI210 smoke path.
- Keep `TP=1`, `DP=4`, and external DP load balancing for the 4-GPU smoke path.
- Disable vLLM sleep/free-cache and torch compile on this native ROCm stack.
- Do not disable remove padding for this smoke path; `USE_REMOVE_PADDING=False` reproduced early silent worker death.
- Treat PPO/GAE critic startup as a separate unresolved issue.
- Check for orphaned vLLM worker subprocesses after completed runs.
