#!/bin/bash

set -euo pipefail

CONDA_ENV_NAME="${CONDA_ENV_NAME:-verlrl}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_VERSION="${TORCH_VERSION:-2.7.1+rocm6.3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/rocm6.3}"
PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx90a}"
VLLM_SOURCE_ROOT="${VLLM_SOURCE_ROOT:-/tmp/vllm-v0.10.0}"
PATCH_VLLM_SOURCE="${PATCH_VLLM_SOURCE:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required but was not found in PATH" >&2
    exit 1
fi

if ! command -v rocminfo >/dev/null 2>&1; then
    echo "rocminfo is required but was not found in PATH" >&2
    exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "${CONDA_ENV_NAME}"; then
    echo "Creating conda env ${CONDA_ENV_NAME} with Python ${PYTHON_VERSION}"
    conda create -y -n "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda activate "${CONDA_ENV_NAME}"

export PYTHONNOUSERSITE=1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES:-1}"
export HSA_FORCE_FINE_GRAIN_PCIE="${HSA_FORCE_FINE_GRAIN_PCIE:-1}"
export RCCL_MSCCL_ENABLE="${RCCL_MSCCL_ENABLE:-0}"
export RCCL_MSCCLPP_ENABLE="${RCCL_MSCCLPP_ENABLE:-0}"
if [ -z "${HIP_VISIBLE_DEVICES:-}" ] && [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
    export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"
fi
if [ -n "${HIP_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}"
    unset ROCR_VISIBLE_DEVICES
fi
export PYTORCH_ROCM_ARCH

echo "Active conda env: ${CONDA_ENV_NAME}"
echo "ROCm arch: ${PYTORCH_ROCM_ARCH}"
rocminfo | grep -E "Marketing Name|Name:" | head -n 12

ACTIVATE_DIR="${CONDA_PREFIX}/etc/conda/activate.d"
DEACTIVATE_DIR="${CONDA_PREFIX}/etc/conda/deactivate.d"
mkdir -p "${ACTIVATE_DIR}" "${DEACTIVATE_DIR}"

cat > "${ACTIVATE_DIR}/verlrl_env.sh" <<'EOF'
#!/bin/bash

_verlrl_store_var() {
    local name="$1"
    local backup_name="_VERLRL_OLD_${name}"
    if [ "${!name+x}" = "x" ]; then
        export "${backup_name}=${!name}"
    else
        export "${backup_name}=__UNSET__"
    fi
}

_verlrl_store_var PYTHONNOUSERSITE
_verlrl_store_var HIP_VISIBLE_DEVICES
_verlrl_store_var ROCR_VISIBLE_DEVICES
_verlrl_store_var CUDA_VISIBLE_DEVICES
_verlrl_store_var RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES
_verlrl_store_var HSA_FORCE_FINE_GRAIN_PCIE
_verlrl_store_var RCCL_MSCCL_ENABLE
_verlrl_store_var RCCL_MSCCLPP_ENABLE

export PYTHONNOUSERSITE=1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export RCCL_MSCCL_ENABLE=0
export RCCL_MSCCLPP_ENABLE=0

if [ -z "${HIP_VISIBLE_DEVICES:-}" ] && [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
    export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"
fi

if [ -n "${HIP_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}"
    unset ROCR_VISIBLE_DEVICES
fi

unset -f _verlrl_store_var
EOF

cat > "${DEACTIVATE_DIR}/verlrl_env.sh" <<'EOF'
#!/bin/bash

_verlrl_restore_var() {
    local name="$1"
    local backup_name="_VERLRL_OLD_${name}"
    local backup_value="${!backup_name-__UNSET__}"
    if [ "${backup_value}" = "__UNSET__" ]; then
        unset "${name}"
    else
        export "${name}=${backup_value}"
    fi
    unset "${backup_name}"
}

_verlrl_restore_var PYTHONNOUSERSITE
_verlrl_restore_var HIP_VISIBLE_DEVICES
_verlrl_restore_var ROCR_VISIBLE_DEVICES
_verlrl_restore_var CUDA_VISIBLE_DEVICES
_verlrl_restore_var RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES
_verlrl_restore_var HSA_FORCE_FINE_GRAIN_PCIE
_verlrl_restore_var RCCL_MSCCL_ENABLE
_verlrl_restore_var RCCL_MSCCLPP_ENABLE

unset -f _verlrl_restore_var
EOF

chmod +x "${ACTIVATE_DIR}/verlrl_env.sh" "${DEACTIVATE_DIR}/verlrl_env.sh"

python -m pip install --upgrade pip setuptools wheel

python -m pip install --upgrade \
    "torch==${TORCH_VERSION}" \
    --index-url "${TORCH_INDEX_URL}"

python -m pip install --upgrade \
    accelerate==1.13.0 \
    codetiming==1.4.0 \
    datasets==4.8.4 \
    diskcache==5.6.3 \
    dill==0.4.1 \
    hydra-core==1.3.2 \
    liger-kernel==0.7.0 \
    lm-format-enforcer==0.11.3 \
    model-hosting-container-standards==0.1.14 \
    "numpy<2.0.0" \
    outlines_core==0.2.11 \
    pandas==3.0.2 \
    peft==0.18.1 \
    pyarrow==23.0.1 \
    pybind11==3.0.3 \
    pylatexenc==2.10 \
    pre-commit==4.5.1 \
    "ray[default]==2.54.1" \
    tensordict==0.10.0 \
    torchdata==0.11.0 \
    tokenizers==0.22.2 \
    transformers==4.57.3 \
    trl==0.9.6 \
    wandb==0.25.1 \
    packaging==25.0 \
    uvicorn==0.44.0 \
    fastapi==0.135.3 \
    latex2sympy2_extended==1.11.0 \
    math_verify==0.9.0 \
    compressed-tensors==0.10.2 \
    tensorizer==2.10.1 \
    runai-model-streamer==0.11.0 \
    runai-model-streamer-s3==0.11.0 \
    tensorboard==2.20.0 \
    xgrammar==0.1.27

python -m pip install -e "${REPO_ROOT}" --no-deps

python - <<'PY'
import importlib.util
import os
from pathlib import Path


def patch_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text()
    if new in text:
        print(f"{label}: already patched")
        return
    if old not in text:
        raise RuntimeError(f"{label}: expected patch target not found in {path}")
    path.write_text(text.replace(old, new))
    print(f"{label}: patched {path}")


compiler_spec = importlib.util.find_spec("triton.compiler.compiler")
if compiler_spec is not None and compiler_spec.origin is not None:
    patch_once(
        Path(compiler_spec.origin),
        "from ..runtime.cache import get_cache_manager, get_dump_manager, get_override_manager, get_cache_key\n",
        """from ..runtime.cache import (
    get_cache_manager,
    get_dump_manager,
    get_override_manager,
    get_cache_key,
    triton_key,
)
""",
        "triton compiler compatibility",
    )
    patch_once(
        Path(compiler_spec.origin),
        "class CompiledKernel:\n\n    def __init__(self, src, metadata_group, hash):\n",
        """class CompiledKernel:
    launch_enter_hook = None
    launch_exit_hook = None

    def __init__(self, src, metadata_group, hash):
""",
        "triton compiled-kernel hook compatibility",
    )
    patch_once(
        Path(compiler_spec.origin),
        """        target = metadata['target']
        metadata['target'] = GPUTarget(target['backend'], target['arch'], target['warp_size'])
        KernelMetadata = namedtuple('KernelMetadata', sorted(list(metadata.keys())))
""",
        """        target = metadata['target']
        metadata['target'] = GPUTarget(target['backend'], target['arch'], target['warp_size'])
        if "cluster_dims" not in metadata:
            metadata["cluster_dims"] = metadata.get("clusterDims", [1, 1, 1])
        KernelMetadata = namedtuple('KernelMetadata', sorted(list(metadata.keys())))
""",
        "triton compiled-kernel metadata compatibility",
    )
    patch_once(
        Path(compiler_spec.origin),
        """        self.hash = hash
        self.name = self.metadata.name
        # stores the text of each level of IR that was generated during compilation
""",
        """        self.hash = hash
        self.name = self.metadata.name
        if hasattr(self.metadata, "num_ctas"):
            self.num_ctas = self.metadata.num_ctas
        self.cluster_dims = tuple(self.metadata.cluster_dims)
        self.clusterDims = self.cluster_dims
        # stores the text of each level of IR that was generated during compilation
""",
        "triton compiled-kernel launcher compatibility",
    )
else:
    print("triton compiler compatibility: skipped (triton.compiler.compiler not found)")

if os.environ.get("PATCH_VLLM_SOURCE", "1") == "1":
    vllm_root = Path(os.environ.get("VLLM_SOURCE_ROOT", "/tmp/vllm-v0.10.0"))
    rocm_path = vllm_root / "vllm/platforms/rocm.py"
    if rocm_path.exists():
        patch_once(
            rocm_path,
            """# Prevent use of clashing `{CUDA/HIP}_VISIBLE_DEVICES`
if "HIP_VISIBLE_DEVICES" in os.environ:
    val = os.environ["HIP_VISIBLE_DEVICES"]
    if cuda_val := os.environ.get("CUDA_VISIBLE_DEVICES", None):
        if val != cuda_val:
            logger.warning(
                "Overriding CUDA_VISIBLE_DEVICES=%s to match HIP_VISIBLE_DEVICES=%s on ROCm",
                cuda_val,
                val,
            )
            os.environ["CUDA_VISIBLE_DEVICES"] = val
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = val
""",
            """# Prevent use of clashing `{CUDA/HIP}_VISIBLE_DEVICES`.
# Ignore an explicitly empty HIP_VISIBLE_DEVICES so zero-GPU Ray actors do not
# erase a usable CUDA_VISIBLE_DEVICES inherited for import-time platform checks.
if "HIP_VISIBLE_DEVICES" in os.environ:
    val = os.environ["HIP_VISIBLE_DEVICES"]
    if val:
        if cuda_val := os.environ.get("CUDA_VISIBLE_DEVICES", None):
            if val != cuda_val:
                logger.warning(
                    "Overriding CUDA_VISIBLE_DEVICES=%s to match HIP_VISIBLE_DEVICES=%s on ROCm",
                    cuda_val,
                    val,
                )
                os.environ["CUDA_VISIBLE_DEVICES"] = val
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = val
""",
            "vLLM ROCm visibility compatibility",
        )
    else:
        print(f"vLLM ROCm visibility compatibility: skipped ({rocm_path} missing)")

    fa_utils_path = vllm_root / "vllm/attention/utils/fa_utils.py"
    if fa_utils_path.exists():
        patch_once(
            fa_utils_path,
            """elif current_platform.is_rocm():
    try:
        from flash_attn import flash_attn_varlen_func  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Rocm platform requires upstream flash-attn to be installed. "
            "Please install flash-attn first."
        ) from e
""",
            """elif current_platform.is_rocm():
    try:
        from flash_attn import flash_attn_varlen_func  # noqa: F401
    except ImportError as e:
        flash_attn_varlen_func = None
        logger.warning(
            "ROCm flash-attn is not installed; continuing without upstream flash-attn support."
        )
""",
            "vLLM ROCm flash-attn compatibility",
        )
    else:
        print(f"vLLM ROCm flash-attn compatibility: skipped ({fa_utils_path} missing)")
PY

python - <<'PY'
import os
import torch
import ray
import verl
import triton.compiler.compiler as triton_compiler

assert torch.cuda.is_available(), "ROCm PyTorch did not detect any GPU"
assert hasattr(triton_compiler, "triton_key"), "triton.compiler.compiler.triton_key is missing"
assert hasattr(
    triton_compiler.CompiledKernel, "launch_enter_hook"
), "triton.compiler.compiler.CompiledKernel.launch_enter_hook is missing"
assert hasattr(
    triton_compiler.CompiledKernel, "launch_exit_hook"
), "triton.compiler.compiler.CompiledKernel.launch_exit_hook is missing"
assert hasattr(
    triton_compiler.CompiledKernel, "cluster_dims"
) or "cluster_dims" in triton_compiler.CompiledKernel.__init__.__code__.co_names, (
    "triton.compiler.compiler.CompiledKernel cluster_dims compatibility patch is missing"
)

probe = torch.randn(8, device="cuda")
print("HIP_VISIBLE_DEVICES", os.environ.get("HIP_VISIBLE_DEVICES"))
print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("ROCR_VISIBLE_DEVICES", os.environ.get("ROCR_VISIBLE_DEVICES"))
print("RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES", os.environ.get("RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES"))
print("HSA_FORCE_FINE_GRAIN_PCIE", os.environ.get("HSA_FORCE_FINE_GRAIN_PCIE"))
print("RCCL_MSCCL_ENABLE", os.environ.get("RCCL_MSCCL_ENABLE"))
print("RCCL_MSCCLPP_ENABLE", os.environ.get("RCCL_MSCCLPP_ENABLE"))
print("torch", torch.__version__)
print("ray", ray.__version__)
print("verl", verl.__version__)
print("triton_key_present", hasattr(triton_compiler, "triton_key"))
print("compiled_kernel_launch_enter_hook_present", hasattr(triton_compiler.CompiledKernel, "launch_enter_hook"))
print("compiled_kernel_launch_exit_hook_present", hasattr(triton_compiler.CompiledKernel, "launch_exit_hook"))
print("compiled_kernel_cluster_dims_patch_present", "cluster_dims" in triton_compiler.CompiledKernel.__init__.__code__.co_names)
print("device", torch.cuda.get_device_name(0))
print("probe_sum", float(probe.sum().cpu()))
PY

python -m verl.trainer.main_ppo --help >/dev/null

echo
echo "AMD core verl install completed successfully in conda env ${CONDA_ENV_NAME}."
echo "This script installs the validated native ROCm path for core verl/FSDP, including PPO critic support via trl."
echo "If you also need ROCm vLLM or SGLang, follow docs/amd_tutorial/amd_build_dockerfile_page.rst."
