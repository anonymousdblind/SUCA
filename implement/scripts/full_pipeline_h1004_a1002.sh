#!/usr/bin/env bash

set -euo pipefail

PROJECT_HOME=/data/yzj/SUCA
ROOT=${PROJECT_HOME}/implement
DATA_ROOT=${PROJECT_HOME}/data
MODEL_ROOT=${PROJECT_HOME}/models
FINE_T2I_ROOT=${DATA_ROOT}/fine-t2i

WARMUP_OUTPUT_DIR=${ROOT}/outputs/warmup
TRAIN_LOG_DIR=${ROOT}/outputs/full_pipeline_logs
RESULT_ROOT=${ROOT}/eval_results/full_pipeline
BASE_RESULT_DIR=${RESULT_ROOT}/base
WARMUP_RESULT_DIR=${RESULT_ROOT}/warmup
RL_RESULT_DIR=${RESULT_ROOT}/rl
COMPARISON_MD=${PROJECT_HOME}/baseline_comparison_auto.md

H100_WARMUP_GPUS=${H100_WARMUP_GPUS:-0,1,2,3}
H100_TRAIN_GPUS=${H100_TRAIN_GPUS:-0,1,2,3}
EVAL_GEN_GPU=${EVAL_GEN_GPU:-0}
EVAL_QWEN_GPU=${EVAL_QWEN_GPU:-0}
LOCAL_REWARD_GPUS=${LOCAL_REWARD_GPUS:-4,5}
USE_REMOTE_REWARD=${USE_REMOTE_REWARD:-1}
A100_HOST=${A100_HOST:-}
WANDB_API_KEY=${WANDB_API_KEY:-2b9b4e9f586c76970ab77b0aded7fc04c909d288}

H100_WARMUP_GPUS=0,1,4,5
H100_TRAIN_GPUS=0,1,4,5
EVAL_GEN_GPU=0
EVAL_QWEN_GPU=2
LOCAL_REWARD_GPUS=2,3
USE_REMOTE_REWARD=0
A100_HOST=localhost

ANALYSIS_ROOT=${ROOT}/analysis
SUMMARY_ROOT=${ANALYSIS_ROOT}/summaries
PAPER_ARTIFACT_DIR=${ROOT}/paper_artifacts
PIPELINE_MANIFEST=${ROOT}/docs/paper_artifact_manifest.canonical.json

DPG_SCORE_COMMAND_TEMPLATE=${DPG_SCORE_COMMAND_TEMPLATE:-}
DPG_VARIANT_NAME_PREFIX=${DPG_VARIANT_NAME_PREFIX:-}

PYTHON_BIN=${ROOT}/.venv/bin/python
TRAIN_LOG=${TRAIN_LOG_DIR}/rl_train.log
TUNNEL_PID=

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

count_csv_items() {
  python3 - <<'PY' "$1"
import sys
items = [x for x in sys.argv[1].split(',') if x.strip()]
print(len(items))
PY
}

wait_for_health() {
  local url="$1"
  local name="$2"
  for _ in $(seq 1 60); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      log "$name is ready: $url"
      return 0
    fi
    sleep 3
  done
  die "$name failed health check: $url"
}

cleanup() {
  local exit_code=$?
  if [[ -n "${TUNNEL_PID}" ]]; then
    kill "${TUNNEL_PID}" 2>/dev/null || true
  fi
  pkill -f "scripts/run_reward_server.py --model" 2>/dev/null || true
  exit ${exit_code}
}

trap cleanup EXIT

require_path() {
  local path="$1"
  [[ -e "$path" ]] || die "Required path not found: $path"
}

read_first_existing_secret() {
  local path
  for path in "$@"; do
    if [[ -f "$path" ]]; then
      head -n 1 "$path" | tr -d '[:space:]'
      return 0
    fi
  done
  return 1
}

resolve_dashscope_key() {
  if [[ -n "${DASHSCOPE_API_KEY:-}" ]]; then
    echo "$DASHSCOPE_API_KEY"
    return 0
  fi

  read_first_existing_secret \
    "$HOME/.dashscope_api_key" \
    "$HOME/.config/dashscope/api_key" \
    "$HOME/.config/dashscope_api_key"
}

prepare_layout() {
  log "Preparing fixed project layout"
  require_path "$ROOT/.venv/bin/activate"
  require_path "$MODEL_ROOT/stable-diffusion-3.5-medium"
  require_path "$MODEL_ROOT/Qwen3-VL-8B-Instruct"
  require_path "$DATA_ROOT/geneval2"
  require_path "$DATA_ROOT/spatialgeneval"
  require_path "$DATA_ROOT/dpg-bench"
  require_path "$FINE_T2I_ROOT"

  mkdir -p "$ROOT/models" "$ROOT/dataset" "$TRAIN_LOG_DIR" "$RESULT_ROOT"
  mkdir -p "$SUMMARY_ROOT/geneval2" "$SUMMARY_ROOT/spatialgeneval" "$SUMMARY_ROOT/dpgbench"
  ln -sfn "$MODEL_ROOT/stable-diffusion-3.5-medium" "$ROOT/models/stable-diffusion-3.5-medium"
  ln -sfn "$MODEL_ROOT/Qwen3-VL-8B-Instruct" "$ROOT/models/Qwen3-VL-8B-Instruct"

  if [[ -d "$DATA_ROOT/t2i_compbench" ]]; then
    ln -sfn "$DATA_ROOT/t2i_compbench" "$ROOT/dataset/t2i_compbench"
  fi
  if [[ -d "$DATA_ROOT/geneval" ]]; then
    ln -sfn "$DATA_ROOT/geneval" "$ROOT/dataset/geneval"
  fi
  if [[ -d "$DATA_ROOT/geneval_compositional" ]]; then
    ln -sfn "$DATA_ROOT/geneval_compositional" "$ROOT/dataset/geneval_compositional"
  fi
}

activate_env() {
  cd "$ROOT"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export NCCL_TIMEOUT=1800
  export PYTHONPATH="$ROOT:$ROOT/flow_grpo:${PYTHONPATH:-}"
  export LD_LIBRARY_PATH="$ROOT/.venv/lib/python3.10/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
}

run_warmup() {
  log "Running 4xH100 warmup"
  activate_env
  local num_processes
  num_processes=$(count_csv_items "$H100_WARMUP_GPUS")
  CUDA_VISIBLE_DEVICES="$H100_WARMUP_GPUS" \
  "$PYTHON_BIN" -m accelerate.commands.launch \
    --num_processes "$num_processes" \
    --mixed_precision bf16 \
    --multi_gpu \
    train_warmup.py \
    --model_name "$MODEL_ROOT/stable-diffusion-3.5-medium" \
    --metadata_file "$FINE_T2I_ROOT/warmup_100k.json" \
    --data_root "$FINE_T2I_ROOT" \
    --output_dir "$WARMUP_OUTPUT_DIR" \
    --resolution 512 \
    --train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --max_train_steps 7500 \
    --gradient_checkpointing \
    --mixed_precision bf16

  require_path "$WARMUP_OUTPUT_DIR/checkpoint-7000/transformer"
}

start_local_reward_servers() {
  log "Starting local reward servers on A100 GPUs ${LOCAL_REWARD_GPUS}"
  activate_env
  mkdir -p "$TRAIN_LOG_DIR"
  pkill -f "scripts/run_reward_server.py --model" 2>/dev/null || true

  local gpu0 gpu1
  gpu0=${LOCAL_REWARD_GPUS%%,*}
  gpu1=${LOCAL_REWARD_GPUS##*,}

  nohup "$PYTHON_BIN" -u scripts/run_reward_server.py \
    --model "$MODEL_ROOT/Qwen3-VL-8B-Instruct" \
    --gpu "$gpu0" \
    --port 8100 \
    > "$TRAIN_LOG_DIR/reward_gpu${gpu0}.log" 2>&1 &

  nohup "$PYTHON_BIN" -u scripts/run_reward_server.py \
    --model "$MODEL_ROOT/Qwen3-VL-8B-Instruct" \
    --gpu "$gpu1" \
    --port 8101 \
    > "$TRAIN_LOG_DIR/reward_gpu${gpu1}.log" 2>&1 &

  wait_for_health "http://127.0.0.1:8100/health" "reward server 8100"
  wait_for_health "http://127.0.0.1:8101/health" "reward server 8101"
}

run_rl_train() {
  log "Running 4xH100 RL training"
  activate_env
  export WANDB_API_KEY
  export SUCA_REWARD_PORTS=8100,8101
  local num_processes
  num_processes=$(count_csv_items "$H100_TRAIN_GPUS")
  CUDA_VISIBLE_DEVICES="$H100_TRAIN_GPUS" \
  "$PYTHON_BIN" -m accelerate.commands.launch \
    --num_processes "$num_processes" \
    --mixed_precision bf16 \
    --multi_gpu \
    flow_grpo/scripts/train_sd3_suca.py \
    --config flow_grpo/config/suca.py:suca_sd3_4gpu \
    2>&1 | tee "$TRAIN_LOG"
}

resolve_rl_transformer() {
  local best_path latest_path
  best_path="$ROOT/logs/suca_compbench/checkpoints/checkpoint-best-ood/transformer"
  if [[ -d "$best_path" ]]; then
    echo "$best_path"
    return 0
  fi

  latest_path=$(find "$ROOT/logs/suca_compbench/checkpoints" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1 || true)
  [[ -n "$latest_path" && -d "$latest_path/transformer" ]] || die "RL checkpoint transformer not found under $ROOT/logs/suca_compbench/checkpoints"
  echo "$latest_path/transformer"
}

generate_variant() {
  local variant="$1"
  local output_dir="$2"
  local transformer_path="${3:-}"
  activate_env

  log "Generating benchmark images for ${variant}"
  if [[ -n "$transformer_path" ]]; then
    CUDA_VISIBLE_DEVICES="$EVAL_GEN_GPU" \
    "$PYTHON_BIN" scripts/generate_images.py \
      --model_path "$MODEL_ROOT/stable-diffusion-3.5-medium" \
      --transformer_path "$transformer_path" \
      --data_dir "$DATA_ROOT" \
      --output_dir "$output_dir" \
      --resolution 512 \
      --batch_size 4 \
      --benchmarks geneval2,spatialgeneval,dpgbench
  else
    CUDA_VISIBLE_DEVICES="$EVAL_GEN_GPU" \
    "$PYTHON_BIN" scripts/generate_images.py \
      --model_path "$MODEL_ROOT/stable-diffusion-3.5-medium" \
      --data_dir "$DATA_ROOT" \
      --output_dir "$output_dir" \
      --resolution 512 \
      --batch_size 4 \
      --benchmarks geneval2,spatialgeneval,dpgbench
  fi
}

run_geneval_eval() {
  local variant="$1"
  local output_dir="$2"
  activate_env
  log "Running GenEval2 for ${variant}"
  CUDA_VISIBLE_DEVICES="$EVAL_QWEN_GPU" \
  "$PYTHON_BIN" scripts/eval_geneval2_shard.py \
    --model_path "$MODEL_ROOT/Qwen3-VL-8B-Instruct" \
    --benchmark_data "$DATA_ROOT/geneval2/geneval2_data.jsonl" \
    --image_paths "$output_dir/geneval2/image_paths.json" \
    --output_file "$output_dir/geneval2/scores.json" \
    --shard_id 0 \
    --num_shards 1
}

run_spatial_eval() {
  local variant="$1"
  local output_dir="$2"
  local dashscope_key
  dashscope_key=$(resolve_dashscope_key || true)
  [[ -n "$dashscope_key" ]] || die "DashScope API key not found. Put it in DASHSCOPE_API_KEY or ~/.dashscope_api_key before running the zero-config pipeline."
  activate_env
  log "Running SpatialGenEval for ${variant}"
  "$PYTHON_BIN" scripts/eval_spatial_api.py \
    --api_key "$dashscope_key" \
    --image_dir "$output_dir/spatialgeneval/images" \
    --input_json "$DATA_ROOT/spatialgeneval/eval/SpatialGenEval_T2I_Prompts.jsonl" \
    --output_json "$output_dir/spatialgeneval/results.json" \
    --summary_json "$output_dir/spatialgeneval/summary.json" \
    --variant_name "$variant" \
    --num_samples 50
}

run_dpg_eval_if_configured() {
  local variant="$1"
  local output_dir="$2"
  if [[ -z "$DPG_SCORE_COMMAND_TEMPLATE" ]]; then
    die "DPG_SCORE_COMMAND_TEMPLATE is not set. Zero-config full paper generation still requires a real DPG scorer backend."
  fi

  activate_env
  log "Running DPG-Bench formal scoring for ${variant}"
  "$PYTHON_BIN" scripts/eval_dpgbench_formal.py \
    --image-dir "$output_dir/dpgbench/images" \
    --prompt-dir "$DATA_ROOT/dpg-bench/dpg_bench/prompts" \
    --raw-score-csv "$output_dir/dpgbench/raw_scores.csv" \
    --summary-json "$output_dir/dpgbench/summary.json" \
    --variant-name "${DPG_VARIANT_NAME_PREFIX}${variant}" \
    --score-command-template "$DPG_SCORE_COMMAND_TEMPLATE"
}

export_pipeline_summaries() {
  log "Exporting normalized summaries to canonical analysis paths"
  activate_env
  "$PYTHON_BIN" scripts/export_pipeline_summaries.py \
    --benchmark-data "$DATA_ROOT/geneval2/geneval2_data.jsonl" \
    --base-dir "$BASE_RESULT_DIR" \
    --warmup-dir "$WARMUP_RESULT_DIR" \
    --rl-dir "$RL_RESULT_DIR" \
    --output-root "$SUMMARY_ROOT"
}

build_paper_artifacts() {
  log "Building paper tables and figures"
  activate_env
  "$PYTHON_BIN" scripts/build_paper_artifacts.py \
    --manifest "$PIPELINE_MANIFEST" \
    --output-dir "$PAPER_ARTIFACT_DIR"
}

write_comparison() {
  log "Writing baseline comparison markdown"
  activate_env
  "$PYTHON_BIN" scripts/write_baseline_comparison.py \
    --base-dir "$BASE_RESULT_DIR" \
    --warmup-dir "$WARMUP_RESULT_DIR" \
    --rl-dir "$RL_RESULT_DIR" \
    --output "$COMPARISON_MD"
}

main() {
  prepare_layout
  run_warmup

  start_local_reward_servers
  run_rl_train

  local warmup_transformer rl_transformer
  warmup_transformer="$WARMUP_OUTPUT_DIR/checkpoint-7000/transformer"
  rl_transformer="$(resolve_rl_transformer)"

  generate_variant base "$BASE_RESULT_DIR"
  generate_variant warmup "$WARMUP_RESULT_DIR" "$warmup_transformer"
  generate_variant rl "$RL_RESULT_DIR" "$rl_transformer"

  run_geneval_eval base "$BASE_RESULT_DIR"
  run_geneval_eval warmup "$WARMUP_RESULT_DIR"
  run_geneval_eval rl "$RL_RESULT_DIR"

  run_spatial_eval base "$BASE_RESULT_DIR"
  run_spatial_eval warmup "$WARMUP_RESULT_DIR"
  run_spatial_eval rl "$RL_RESULT_DIR"

  run_dpg_eval_if_configured base "$BASE_RESULT_DIR"
  run_dpg_eval_if_configured warmup "$WARMUP_RESULT_DIR"
  run_dpg_eval_if_configured rl "$RL_RESULT_DIR"

  export_pipeline_summaries
  build_paper_artifacts
  write_comparison
  log "Full pipeline finished. Comparison file: $COMPARISON_MD"
  log "Paper artifacts: $PAPER_ARTIFACT_DIR"
}

main "$@"