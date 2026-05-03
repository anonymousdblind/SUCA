#!/bin/bash
# Run evaluation for base and SFT models in parallel
# Base model on GPU 2,3 | SFT model on GPU 6,7

PROJECT_DIR="/tcci_mnt/liguoyi/project/SUGA0316"
PYTHON="${PROJECT_DIR}/.venv/bin/python"
RESULT_DIR="${PROJECT_DIR}/eval_results"
mkdir -p "${RESULT_DIR}"

echo "=========================================="
echo "Starting evaluation pipeline"
echo "Base model: GPU 2 | SFT model: GPU 6"
echo "=========================================="

# Phase 1: Generate images (in parallel)
echo ""
echo "[Phase 1] Generating images..."

# Base model generation on GPU 2
echo "  Starting base model generation on GPU 2..."
CUDA_VISIBLE_DEVICES=2 nohup ${PYTHON} ${PROJECT_DIR}/scripts/generate_images.py \
    --model_path ${PROJECT_DIR}/models/stable-diffusion-3.5-medium \
    --data_dir ${PROJECT_DIR}/data \
    --output_dir ${RESULT_DIR}/base \
    --resolution 512 \
    --batch_size 4 \
    --benchmarks geneval2,spatialgeneval,dpgbench \
    > ${RESULT_DIR}/gen_base.log 2>&1 &
BASE_GEN_PID=$!

# SFT model generation on GPU 6
echo "  Starting SFT model generation on GPU 6..."
CUDA_VISIBLE_DEVICES=6 nohup ${PYTHON} ${PROJECT_DIR}/scripts/generate_images.py \
    --model_path ${PROJECT_DIR}/models/stable-diffusion-3.5-medium \
    --transformer_path ${PROJECT_DIR}/outputs/warmup/checkpoint-7000/transformer \
    --data_dir ${PROJECT_DIR}/data \
    --output_dir ${RESULT_DIR}/sft \
    --resolution 512 \
    --batch_size 4 \
    --benchmarks geneval2,spatialgeneval,dpgbench \
    > ${RESULT_DIR}/gen_sft.log 2>&1 &
SFT_GEN_PID=$!

echo "  Base PID: ${BASE_GEN_PID}, SFT PID: ${SFT_GEN_PID}"
echo "  Waiting for image generation to complete..."
wait ${BASE_GEN_PID}
BASE_GEN_EXIT=$?
echo "  Base model generation done (exit: ${BASE_GEN_EXIT})"
wait ${SFT_GEN_PID}
SFT_GEN_EXIT=$?
echo "  SFT model generation done (exit: ${SFT_GEN_EXIT})"

if [ ${BASE_GEN_EXIT} -ne 0 ] || [ ${SFT_GEN_EXIT} -ne 0 ]; then
    echo "ERROR: Image generation failed. Check logs."
    exit 1
fi

# Phase 2: Run GenEval2 evaluation (in parallel)
echo ""
echo "[Phase 2] Running GenEval2 evaluation..."

# Modify evaluation.py to accept model_path argument - use our local Qwen3-VL
QWEN_MODEL="${PROJECT_DIR}/models/Qwen3-VL-8B-Instruct"

# Base model eval on GPU 3
echo "  Running GenEval2 eval for base model on GPU 3..."
CUDA_VISIBLE_DEVICES=3 nohup ${PYTHON} -c "
import os, sys, json, torch
from tqdm import tqdm
from scipy.stats import gmean

# Patch model path before importing evaluation
os.environ['QWEN_MODEL_PATH'] = '${QWEN_MODEL}'

sys.path.insert(0, '${PROJECT_DIR}/data/geneval2')
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

print('Loading Qwen3-VL...')
processor = AutoProcessor.from_pretrained('${QWEN_MODEL}', torch_dtype='auto')
model = Qwen3VLForConditionalGeneration.from_pretrained('${QWEN_MODEL}', dtype='auto', device_map='auto')

def send_message_with_image(prompt, image_filepath, answer_list=None):
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image_filepath}, {'type': 'text', 'text': prompt}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt')
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
    mapping = {'one':'1','two':'2','three':'3','four':'4','five':'5','six':'6','seven':'7','eight':'8','nine':'9','ten':'10'}
    return mapping.get(number, 'other')

benchmark_data = [json.loads(l) for l in open('${PROJECT_DIR}/data/geneval2/geneval2_data.jsonl')]
image_data = json.load(open('${RESULT_DIR}/base/geneval2/image_paths.json'))

all_score_lists = []
for d in tqdm(benchmark_data, desc='GenEval2-base'):
    image_filepath = image_data[d['prompt']]
    score_list = []
    for vqa in d['vqa_list']:
        question, answer = vqa
        if question.startswith('How many'):
            answer_list = [answer, answer.capitalize(), ' '+answer, ' '+answer.capitalize(), return_numeric_string(answer), ' '+return_numeric_string(answer)]
        else:
            answer_list = ['Yes', 'yes', ' yes', ' Yes']
        pred, ans_prob = send_message_with_image('{} Answer in one word.'.format(question), image_filepath, answer_list=answer_list)
        score_list.append(ans_prob)
    all_score_lists.append(score_list)

json.dump(all_score_lists, open('${RESULT_DIR}/base/geneval2/scores.json', 'w'))
per_prompt = [gmean(s) for s in all_score_lists]
total = 100 * sum(per_prompt) / len(per_prompt)
print(f'GenEval2 Base Score (soft_tifa_gm): {total:.2f}')
with open('${RESULT_DIR}/base/geneval2/result.txt', 'w') as f:
    f.write(f'soft_tifa_gm: {total:.2f}\n')
" > ${RESULT_DIR}/eval_geneval2_base.log 2>&1 &
GENEVAL_BASE_PID=$!

# SFT model eval on GPU 7
echo "  Running GenEval2 eval for SFT model on GPU 7..."
CUDA_VISIBLE_DEVICES=7 nohup ${PYTHON} -c "
import os, sys, json, torch
from tqdm import tqdm
from scipy.stats import gmean

sys.path.insert(0, '${PROJECT_DIR}/data/geneval2')
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

print('Loading Qwen3-VL...')
processor = AutoProcessor.from_pretrained('${QWEN_MODEL}', torch_dtype='auto')
model = Qwen3VLForConditionalGeneration.from_pretrained('${QWEN_MODEL}', dtype='auto', device_map='auto')

def send_message_with_image(prompt, image_filepath, answer_list=None):
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image_filepath}, {'type': 'text', 'text': prompt}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt')
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
    mapping = {'one':'1','two':'2','three':'3','four':'4','five':'5','six':'6','seven':'7','eight':'8','nine':'9','ten':'10'}
    return mapping.get(number, 'other')

benchmark_data = [json.loads(l) for l in open('${PROJECT_DIR}/data/geneval2/geneval2_data.jsonl')]
image_data = json.load(open('${RESULT_DIR}/sft/geneval2/image_paths.json'))

all_score_lists = []
for d in tqdm(benchmark_data, desc='GenEval2-sft'):
    image_filepath = image_data[d['prompt']]
    score_list = []
    for vqa in d['vqa_list']:
        question, answer = vqa
        if question.startswith('How many'):
            answer_list = [answer, answer.capitalize(), ' '+answer, ' '+answer.capitalize(), return_numeric_string(answer), ' '+return_numeric_string(answer)]
        else:
            answer_list = ['Yes', 'yes', ' yes', ' Yes']
        pred, ans_prob = send_message_with_image('{} Answer in one word.'.format(question), image_filepath, answer_list=answer_list)
        score_list.append(ans_prob)
    all_score_lists.append(score_list)

json.dump(all_score_lists, open('${RESULT_DIR}/sft/geneval2/scores.json', 'w'))
per_prompt = [gmean(s) for s in all_score_lists]
total = 100 * sum(per_prompt) / len(per_prompt)
print(f'GenEval2 SFT Score (soft_tifa_gm): {total:.2f}')
with open('${RESULT_DIR}/sft/geneval2/result.txt', 'w') as f:
    f.write(f'soft_tifa_gm: {total:.2f}\n')
" > ${RESULT_DIR}/eval_geneval2_sft.log 2>&1 &
GENEVAL_SFT_PID=$!

echo "  GenEval2 Base PID: ${GENEVAL_BASE_PID}, SFT PID: ${GENEVAL_SFT_PID}"
echo "  Waiting..."
wait ${GENEVAL_BASE_PID}
wait ${GENEVAL_SFT_PID}
echo "  GenEval2 evaluation done."

# Phase 3: Collect results
echo ""
echo "=========================================="
echo "[Results Summary]"
echo "=========================================="
echo ""
echo "--- GenEval2 ---"
echo "Base: $(cat ${RESULT_DIR}/base/geneval2/result.txt 2>/dev/null || echo 'N/A')"
echo "SFT:  $(cat ${RESULT_DIR}/sft/geneval2/result.txt 2>/dev/null || echo 'N/A')"
echo ""
echo "--- SpatialGenEval ---"
echo "Note: Requires Qwen2.5-VL-72B for evaluation (too large for current setup)"
echo "Images generated at: ${RESULT_DIR}/{base,sft}/spatialgeneval/images/"
echo ""
echo "--- DPG-Bench ---"
echo "Note: Requires mPLUG VQA model for evaluation"
echo "Images generated at: ${RESULT_DIR}/{base,sft}/dpgbench/images/"
echo ""

# Save summary
cat > ${RESULT_DIR}/summary.txt << 'SUMMARY'
=== Evaluation Summary ===
Model: SD3.5-Medium
Resolution: 512
Benchmarks: GenEval2, SpatialGenEval, DPG-Bench

--- GenEval2 (soft_tifa_gm) ---
SUMMARY
echo "Base: $(cat ${RESULT_DIR}/base/geneval2/result.txt 2>/dev/null || echo 'N/A')" >> ${RESULT_DIR}/summary.txt
echo "SFT:  $(cat ${RESULT_DIR}/sft/geneval2/result.txt 2>/dev/null || echo 'N/A')" >> ${RESULT_DIR}/summary.txt

echo ""
echo "Full summary saved to: ${RESULT_DIR}/summary.txt"
echo "Done!"
