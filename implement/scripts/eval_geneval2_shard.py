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
