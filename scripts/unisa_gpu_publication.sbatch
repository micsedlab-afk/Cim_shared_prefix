#!/bin/bash
#SBATCH --job-name=cim-gpu-qualification
#SBATCH --partition=aiq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00

# UNISA H100 reproducibility entrypoint. This script is committed but has
# NOT been submitted: the GitHub connector does not offer SSH/sbatch access.
# Never write H100 latency from CPU-only process, or overwrite old acquisitions.
set -euo pipefail

: "${CIM_REPO_ROOT:?absolute checkout of Cim_shared_prefix required}"
: "${CIM_TASK:?attention|decoder|quality_qualify|quality_calibrate|quality_evaluate}"
CIM_MODEL="${CIM_MODEL:-llama}"
CIM_PYTHON="${CIM_PYTHON:-python}"
CIM_GPU_RESULTS_ROOT="${CIM_GPU_RESULTS_ROOT:-$HOME/H100calib/cim_shared_prefix_gpu_20261104}"
CIM_PREFIX="${CIM_PREFIX:-2048}"
CIM_SUFFIX="${CIM_SUFFIX:-128}"
CIM_BATCH="${CIM_BATCH:-1}"
CIM_OUTPUT_TOKENS="${CIM_OUTPUT_TOKENS:-128}"
CIM_PRECISION="${CIM_PRECISION:-bf16}"
CIM_RUN_ID="${SLURM_JOB_ID:?Slurm allocation missing}"
cd "$CIM_REPO_ROOT"
export PYTHONPATH="$CIM_REPO_ROOT/src:${PYTHONPATH:-}"
mkdir -p "$CIM_GPU_RESULTS_ROOT/logs"
OUT="$CIM_GPU_RESULTS_ROOT/${CIM_MODEL}/${CIM_TASK}/job_${CIM_RUN_ID}"
mkdir -p "$OUT"

"$CIM_PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("NO_HARDWARE_GPU: no simulated/CPU benchmark accepted")
p=torch.cuda.get_device_properties(0)
print({"name":p.name,"compute_capability":(p.major,p.minor),
       "torch":torch.__version__,"CUDA_runtime":torch.version.cuda},flush=True)
PY
nvidia-smi --query-gpu=name,uuid,pci.bus_id,driver_version --format=csv,noheader >"$OUT/gpu_identity.csv"

case "$CIM_TASK" in
  attention)
    "$CIM_PYTHON" scripts/benchmark_gpu_attention.py \
      --model "$CIM_MODEL" --prefix "$CIM_PREFIX" --suffix "$CIM_SUFFIX" \
      --batch "$CIM_BATCH" --output-tokens "$CIM_OUTPUT_TOKENS" \
      --precision "$CIM_PRECISION" --steps all \
      --modes graph --candidates grouped_gemm,compiled_grouped_gemm \
      --out "$OUT/attention_cuda_events.json"
    ;;
  decoder)
    : "${CIM_MODEL_SNAPSHOT:?exact revision-pinned safetensors snapshot required}"
    "$CIM_PYTHON" scripts/benchmark_real_decoder.py \
      --model "$CIM_MODEL" --model-root "$CIM_MODEL_SNAPSHOT" \
      --precision "$CIM_PRECISION" --prefix "$CIM_PREFIX" \
      --suffix "$CIM_SUFFIX" --batch "$CIM_BATCH" \
      --output-tokens "$CIM_OUTPUT_TOKENS" --rounds 5 \
      --profile-modules --out "$OUT/full_pretrained_decoder.json"
    ;;
  quality_qualify|quality_calibrate|quality_evaluate)
    if [[ "$CIM_MODEL" != "opt" && "$CIM_MODEL" != "mistral" ]]; then
      echo "R0-R3 extensions apply only to OPT/Mistral" >&2; exit 2
    fi
    : "${CIM_MODEL_SNAPSHOT:?exact revision-pinned full pretrained snapshot required}"
    : "${CIM_UPSTREAM_ROOT:?source-verified SRAM2LLM/shared_prefix_cim root required}"
    # Qualified full model quality requires BF16 irrespective of native FP16
    # acquisition elsewhere; OPT BF16 remains a separate evaluation variant.
    MODE="${CIM_TASK#quality_}"
    "$CIM_PYTHON" scripts/run_extended_quality.py \
      --model "$CIM_MODEL" --mode "$MODE" \
      --model-root "$CIM_MODEL_SNAPSHOT" \
      --upstream-root "$CIM_UPSTREAM_ROOT" \
      --work-root "$CIM_GPU_RESULTS_ROOT/quality"
    ;;
  *)
    echo "UNKNOWN_CIM_TASK: $CIM_TASK" >&2; exit 2
    ;;
esac
printf '%s\n' "COMPLETED_REAL_GPU_TASK=$CIM_TASK MODEL=$CIM_MODEL JOB=$CIM_RUN_ID"
