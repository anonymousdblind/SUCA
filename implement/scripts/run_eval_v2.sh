#!/bin/bash
# Evaluation v2: maximize GPU utilization
# Phase 1: 4 GPUs all generating images in parallel
# Phase 2: 4 GPUs all running evaluation in parallel

PROJECT_DIR="/tcci_mnt/liguoyi/project/SUGA0316"
PYTHON="${PROJECT_DIR}/.venv/bin/python"
RESULT_DIR="${PROJECT_DIR}/eval_results"
QWEN_MODEL="${PROJECT_DIR}/models/Qwen3-VL-8B-Instruct"
mkdir -p "${RESULT_DIR}"

echo "=========================================="
echo "Evaluation Pipeline v2 - Full GPU Utilization"
echo "=========================================="

# ==========================================
# Phase 1: Generate images (4 GPUs parallel)
# GPU 2: base - remaining geneval2 + spatialgeneval
# GPU 3: base - dpgbench
# GPU 6: sft - remaining geneval2 + spatialgeneval
# GPU 7: sft - dpgbench
# ==========================================
echo ""
echo "[Phase 1] Generating images on 4 GPUs..."

# Base: geneval2 (remaining) + spatialgeneval on GPU 2
CUDA_VISIBLE_DEVICES=2 ${PYTHON} ${PROJECT_DIR}/scripts/generate_images.py \
    --model_path ${PROJECT_DIR}/models/stable-diffusion-3.5-medium \
    --data_dir ${PROJECT_DIR}/data \
    --output_dir ${RESULT_DIR}/base \
    --resolution 512 --batch_size 4 \
    --benchmarks geneval2,spatialgeneval \
    > ${RESULT_DIR}/gen_base_1.log 2>&1 &
PID1=$!

# Base: dpgbench on GPU 3
CUDA_VISIBLE_DEVICES=3 ${PYTHON} ${PROJECT_DIR}/scripts/generate_images.py \
    --model_path ${PROJECT_DIR}/models/stable-diffusion-3.5-medium \
    --data_dir ${PROJECT_DIR}/data \
    --output_dir ${RESULT_DIR}/base \
    --resolution 512 --batch_size 4 \
    --benchmarks dpgbench \
    > ${RESULT_DIR}/gen_base_2.log 2>&1 &
PID2=$!

# SFT: geneval2 (remaining) + spatialgeneval on GPU 6
CUDA_VISIBLE_DEVICES=6 ${PYTHON} ${PROJECT_DIR}/scripts/generate_images.py \
    --model_path ${PROJECT_DIR}/models/stable-diffusion-3.5-medium \
    --transformer_path ${PROJECT_DIR}/outputs/warmup/checkpoint-7000/transformer \
    --data_dir ${PROJECT_DIR}/data \
    --output_dir ${RESULT_DIR}/sft \
    --resolution 512 --batch_size 4 \
    --benchmarks geneval2,spatialgeneval \
    > ${RESULT_DIR}/gen_sft_1.log 2>&1 &
PID3=$!

# SFT: dpgbench on GPU 7
CUDA_VISIBLE_DEVICES=7 ${PYTHON} ${PROJECT_DIR}/scripts/generate_images.py \
    --model_path ${PROJECT_DIR}/models/stable-diffusion-3.5-medium \
    --transformer_path ${PROJECT_DIR}/outputs/warmup/checkpoint-7000/transformer \
    --data_dir ${PROJECT_DIR}/data \
    --output_dir ${RESULT_DIR}/sft \
    --resolution 512 --batch_size 4 \
    --benchmarks dpgbench \
    > ${RESULT_DIR}/gen_sft_2.log 2>&1 &
PID4=$!

echo "  GPU 2 (base geneval2+spatial): PID ${PID1}"
echo "  GPU 3 (base dpgbench):         PID ${PID2}"
echo "  GPU 6 (sft geneval2+spatial):   PID ${PID3}"
echo "  GPU 7 (sft dpgbench):           PID ${PID4}"
echo "  Waiting for all generation to complete..."

wait ${PID1} ${PID2} ${PID3} ${PID4}
echo "[Phase 1] All image generation done!"

# ==========================================
# Phase 2: Run GenEval2 evaluation (4 GPUs parallel)
# Split 800 prompts into 4 shards, 2 for base, 2 for sft
# ==========================================
echo ""
echo "[Phase 2] Running GenEval2 evaluation on 4 GPUs..."

# Create evaluation script that accepts shard args
cat > ${PROJECT_DIR}/scripts/eval_geneval2_shard.py << 'EVALEOF'
import os, sys, json, torch, argparse
from tqdm import tqdm
from scipy.stats import gmean
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--benchmark_data", type=str, required=True)
parser.add_argument("--image_paths", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--shard_id", type=int, default=0)
parser.add_argument("--num_shards", type=int, default=1)
args = parser.parse_args()

print(f"Loading Qwen3-VL from {args.model_path}...")
processor = AutoProcessor.from_pretrained(args.model_path, torch_dtype="auto")
model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, dtype="auto", device_map="auto")

def send_message_with_image(prompt, image_filepath, answer_list=None):
    messages = [{"role": "user", "content": [{"type": "image", "image": image_filepath}, {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=1, do_sample=False, output_scores=True, return_dict_in_generate=True)
    scores = outputs.scores[0]
    probs = torch.nn.functional.softmax(scores, dim=-1)
    if answer_list:
        lm_prob = sum(probs[0, processor.tokenizer.encode(a)[0]].item() for a in answer_list)
    else:
        lm_prob = None
    pred = processor.batch_decode([torch.argmax(probs)])[0]
    return pred, lm_prob

def return_numeric_string(number):
    mapping = {"one":"1","two":"2","three":"3","four":"4","five":"5","six":"6","seven":"7","eight":"8","nine":"9","ten":"10"}
    return mapping.get(number, "other")

benchmark_data = [json.loads(l) for l in open(args.benchmark_data)]
image_data = json.load(open(args.image_paths))

# Shard the data
n = len(benchmark_data)
shard_size = (n + args.num_shards - 1) // args.num_shards
start = args.shard_id * shard_size
end = min(start + shard_size, n)
my_data = benchmark_data[start:end]
print(f"Shard {args.shard_id}/{args.num_shards}: processing prompts {start}-{end} ({len(my_data)} items)")

all_score_lists = []
for d in tqdm(my_data, desc=f"shard-{args.shard_id}"):
    image_filepath = image_data[d["prompt"]]
    score_list = []
    for vqa in d["vqa_list"]:
        question, answer = vqa
        if question.startswith("How many"):
            answer_list = [answer, answer.capitalize(), " "+answer, " "+answer.capitalize(), return_numeric_string(answer), " "+return_numeric_string(answer)]
        else:
            answer_list = ["Yes", "yes", " yes", " Yes"]
        pred, ans_prob = send_message_with_image("{} Answer in one word.".format(question), image_filepath, answer_list=answer_list)
        score_list.append(ans_prob)
    all_score_lists.append(score_list)

json.dump(all_score_lists, open(args.output_file, "w"))
per_prompt = [gmean(s) for s in all_score_lists]
total = 100 * sum(per_prompt) / len(per_prompt)
print(f"Shard {args.shard_id} Score: {total:.2f}")
EVALEOF

# Run 4 shards in parallel: 2 for base, 2 for sft
CUDA_VISIBLE_DEVICES=2 ${PYTHON} ${PROJECT_DIR}/scripts/eval_geneval2_shard.py \
    --model_path ${QWEN_MODEL} \
    --benchmark_data ${PROJECT_DIR}/data/geneval2/geneval2_data.jsonl \
    --image_paths ${RESULT_DIR}/base/geneval2/image_paths.json \
    --output_file ${RESULT_DIR}/base/geneval2/scores_0.json \
    --shard_id 0 --num_shards 2 \
    > ${RESULT_DIR}/eval_geneval2_base_0.log 2>&1 &
EPID1=$!

CUDA_VISIBLE_DEVICES=3 ${PYTHON} ${PROJECT_DIR}/scripts/eval_geneval2_shard.py \
    --model_path ${QWEN_MODEL} \
    --benchmark_data ${PROJECT_DIR}/data/geneval2/geneval2_data.jsonl \
    --image_paths ${RESULT_DIR}/base/geneval2/image_paths.json \
    --output_file ${RESULT_DIR}/base/geneval2/scores_1.json \
    --shard_id 1 --num_shards 2 \
    > ${RESULT_DIR}/eval_geneval2_base_1.log 2>&1 &
EPID2=$!

CUDA_VISIBLE_DEVICES=6 ${PYTHON} ${PROJECT_DIR}/scripts/eval_geneval2_shard.py \
    --model_path ${QWEN_MODEL} \
    --benchmark_data ${PROJECT_DIR}/data/geneval2/geneval2_data.jsonl \
    --image_paths ${RESULT_DIR}/sft/geneval2/image_paths.json \
    --output_file ${RESULT_DIR}/sft/geneval2/scores_0.json \
    --shard_id 0 --num_shards 2 \
    > ${RESULT_DIR}/eval_geneval2_sft_0.log 2>&1 &
EPID3=$!

CUDA_VISIBLE_DEVICES=7 ${PYTHON} ${PROJECT_DIR}/scripts/eval_geneval2_shard.py \
    --model_path ${QWEN_MODEL} \
    --benchmark_data ${PROJECT_DIR}/data/geneval2/geneval2_data.jsonl \
    --image_paths ${RESULT_DIR}/sft/geneval2/image_paths.json \
    --output_file ${RESULT_DIR}/sft/geneval2/scores_1.json \
    --shard_id 1 --num_shards 2 \
    > ${RESULT_DIR}/eval_geneval2_sft_1.log 2>&1 &
EPID4=$!

echo "  GPU 2 (base shard 0): PID ${EPID1}"
echo "  GPU 3 (base shard 1): PID ${EPID2}"
echo "  GPU 6 (sft shard 0):  PID ${EPID3}"
echo "  GPU 7 (sft shard 1):  PID ${EPID4}"
echo "  Waiting for evaluation..."

wait ${EPID1} ${EPID2} ${EPID3} ${EPID4}
echo "[Phase 2] GenEval2 evaluation done!"

# ==========================================
# Phase 3: Merge results and summarize
# ==========================================
echo ""
echo "[Phase 3] Merging results..."

${PYTHON} -c "
import json
from scipy.stats import gmean

for model in ['base', 'sft']:
    all_scores = []
    for shard in range(2):
        path = 'eval_results/${model}/geneval2/scores_${shard}.json'
        try:
            scores = json.load(open(path.replace('\${model}', model).replace('\${shard}', str(shard))))
            all_scores.extend(scores)
        except:
            pass

    if all_scores:
        per_prompt = [gmean(s) for s in all_scores]
        total = 100 * sum(per_prompt) / len(per_prompt)
        print(f'GenEval2 {model}: {total:.2f} (n={len(all_scores)})')
        with open(f'eval_results/{model}/geneval2/result.txt', 'w') as f:
            f.write(f'soft_tifa_gm: {total:.2f}\n')
            f.write(f'num_prompts: {len(all_scores)}\n')
" 2>&1

# Final summary
echo ""
echo "=========================================="
echo "[Final Results]"
echo "=========================================="
echo "--- GenEval2 (soft_tifa_gm) ---"
echo "Base: $(cat ${RESULT_DIR}/base/geneval2/result.txt 2>/dev/null || echo 'N/A')"
echo "SFT:  $(cat ${RESULT_DIR}/sft/geneval2/result.txt 2>/dev/null || echo 'N/A')"
echo ""
echo "--- Images Generated ---"
for model in base sft; do
    for bench in geneval2 spatialgeneval dpgbench; do
        count=$(ls ${RESULT_DIR}/${model}/${bench}/images/ 2>/dev/null | wc -l)
        echo "  ${model}/${bench}: ${count} images"
    done
done
echo ""
echo "Done!"
