# Troubleshooting Notes

This file records local debugging lessons that are not yet broad project documentation.

## Native MI210 vLLM Smoke Path

| Symptom | Likely Cause | Fix |
|---|---|---|
| Run completes but MI210 VRAM remains allocated | Orphaned vLLM multiprocessing workers survived Ray shutdown | Run `rocm-smi --showpids --showmemuse`, inspect stale worker `ps` lines, then kill only confirmed stale vLLM parent/child processes |
| `ModuleNotFoundError: No module named 'flash_attn'` during old log-prob | Native ROCm env lacks CUDA `flash_attn.bert_padding` | Use the `verl/utils/attention_utils.py` Hugging Face padding fallback and keep `USE_REMOVE_PADDING=True` |
| `Sleep mode is not supported on current platform` | ROCm vLLM does not support sleep mode on this stack | Set `actor_rollout_ref.rollout.free_cache_engine=False` and `+actor_rollout_ref.rollout.enable_sleep_mode=False` |
| `msgspec.ValidationError` around `parallel_config` with `None` | External-DP master port was not propagated to every vLLM server | Pass `data_parallel_master_port` from rank 0 to every external-DP vLLM server before engine config creation |
| Ray `ActorDiedError` during `trainer.init_workers()` with GAE | PPO/critic path is still unstable on the native MI210 stack | Use `ADV_ESTIMATOR=grpo` for smoke runs and debug critic initialization separately |
