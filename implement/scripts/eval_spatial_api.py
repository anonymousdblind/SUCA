"""Run SpatialGenEval on 50 samples using Qwen2.5-VL-72B via DashScope API."""
import os
import json
import base64
import argparse
import time
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


RELATION_FAMILIES = {
    "Left/Right": ["left", "right", "left/right", "right/left"],
    "Above/Below": ["above", "below", "top", "under", "up/down"],
    "Inside/Around": ["inside", "outside", "around", "surround", "contain"],
    "Front/Behind": ["front", "behind", "back", "in front of"],
}

# Reuse the template from SpatialGenEval
vlm_content_template = '''
### Task Description:
You are tasked with carefully examining the provided image and answering the following 10 multiple-choice questions. You MUST ONLY rely on the provided image to answer the questions. DO NOT use any external resources like world knowledge or external information beyond the provided image.

### Multiple-Choice Questions:
##Multiple-Choice Questions##

### Instructions:
1. Answer these 10 questions on a separate 10 lines, beginning with the correct choice option (A/B/C/D/E/..., not the number) and followed by a detailed reason (in the same line as answer).
2. Maintain the exact order of the questions in your answers.
3. Provide only one answer per question.
4. Each answer must be on its own line.
5. Ensure the index of answers matches the index of questions.
6. Select the option 'E: None' when the image can not answer the question.

### Output Format (Example, 10 lines for 10 questions):
E: None - The image does not depict a log or any specific object categories clearly enough to match any listed options.
B: Large and brown bear, small and red fox - The bear is visibly larger and brown, while the fox is smaller and red.
C: The bear is on the left and the fox is on the right - The bear appears on the left and the fox on the right side of the image.
A: The bear is facing the fox - The bear is looking directly at the fox, indicating it is facing the fox.
B: They are positioned opposite each other on the left and right - They are facing each other from opposite sides of the image.
E: None - The image does not provide clear indication of height comparison that matches the provided statements.
B: They are positioned closely together - Bear and fox are seen near each other, interacting without any major distance or separation.
E: None - The image does not show any notable occlusion from logs or surrounding objects.
E: None - The image does not show the bear initiating any of the described motions.
E: None - No direct causal results of the bear's movement are depicted in the image.
'''


def format_questions_prompt(questions):
    question_texts = [item.strip() for item in questions]
    formatted_questions = "\n".join(question_texts)
    return vlm_content_template.replace("##Multiple-Choice Questions##", formatted_questions)


def api_call(client, vlm_prompt, image_path, model_name, temperature=1.0):
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    messages = [
        {"role": "system", "content": "You are a professional image critic."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": vlm_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
            ]
        }
    ]

    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
    )
    return completion.choices[0].message.content


def check_qa(answers, all_preds, min_count=3):
    num_questions = len(answers)
    results = []
    selected_options = []

    for q_idx in range(num_questions):
        option_count = {}
        valid_preds = [preds[q_idx] for preds in all_preds if len(preds) > q_idx]
        if not valid_preds:
            results.append(False)
            selected_options.append(["(No valid predictions)"])
            continue

        for option in valid_preds:
            option_count[option] = option_count.get(option, 0) + 1

        high_freq = [opt for opt, count in option_count.items() if count >= min_count]
        is_correct = len(high_freq) > 0 and answers[q_idx] in high_freq
        results.append(is_correct)

        if high_freq:
            selected_options.append(high_freq)
        else:
            max_count = max(option_count.values())
            most_frequent = [opt for opt, count in option_count.items() if count == max_count]
            selected_options.append(most_frequent)

    return results, selected_options


def process_item(idx, data, image_path, client, model_name, rollout=5, count=4):
    thread_id = threading.get_ident()
    questions = data.get('questions', [])
    answers = data.get('answers', [])

    if not os.path.exists(image_path):
        print(f"  [Skip] Image not found: {image_path}")
        return None

    vlm_prompt = format_questions_prompt(questions)

    model_preds_list = []
    model_preds_cot_list = []
    max_attempts = rollout * 5
    total_attempts = 0

    while len(model_preds_list) < rollout and total_attempts < max_attempts:
        total_attempts += 1
        try:
            raw = api_call(client, vlm_prompt, image_path, model_name)
            if not raw:
                time.sleep(2)
                continue

            preds_cot = [line.strip() for line in raw.strip().split('\n') if line.strip()]
            if len(preds_cot) == len(questions):
                preds = [cot[0] for cot in preds_cot]
                model_preds_list.append(preds)
                model_preds_cot_list.append(preds_cot)
            else:
                print(f"  [Warning] ID {data['id']} format mismatch: got {len(preds_cot)} lines, expected {len(questions)}")
        except Exception as e:
            print(f"  [Error] ID {data['id']}: {e}")
            time.sleep(3)

    if len(model_preds_list) < rollout:
        print(f"  [Failed] ID {data['id']}: only {len(model_preds_list)}/{rollout} valid rollouts")
        return None

    results, selected_options = check_qa(answers, model_preds_list, count)

    return {
        "id": data["id"],
        "scene": data.get("scene", ""),
        "avg_acc": f"{sum(results)}/{len(results)}",
        "basic_acc": f"{sum(results[:2])}/{len(results[:2])}",
        "spatial_acc": f"{sum(results[2:])}/{len(results[2:])}",
        "prompt": data["prompt"],
        "answers": answers,
        "model_preds": selected_options,
        "true-or-false": results,
    }


def detect_relation_family(record):
    texts = [
        str(record.get("scene", "")),
        str(record.get("prompt", "")),
        " ".join(record.get("answers", [])),
    ]
    normalized = " ".join(texts).lower()
    for family, aliases in RELATION_FAMILIES.items():
        if any(alias in normalized for alias in aliases):
            return family
    return None


def build_paper_summary(results, variant_name=None):
    avg_accs = [sum(r['true-or-false']) / len(r['true-or-false']) for r in results]
    basic_accs = [sum(r['true-or-false'][:2]) / 2 for r in results]
    spatial_accs = [sum(r['true-or-false'][2:]) / 8 for r in results]

    grouped = {family: [] for family in RELATION_FAMILIES}
    for item in results:
        family = detect_relation_family(item)
        if family is None:
            continue
        grouped[family].append(sum(item['true-or-false']) / len(item['true-or-false']))

    summary = {
        "benchmark": "spatialgeneval",
        "overall": {
            "avg_acc": 100 * sum(avg_accs) / len(avg_accs),
            "basic_acc": 100 * sum(basic_accs) / len(basic_accs),
            "spatial_acc": 100 * sum(spatial_accs) / len(spatial_accs),
        },
        "dimensions": {},
        "num_samples": len(results),
    }
    if variant_name:
        summary["variant"] = variant_name

    for family, values in grouped.items():
        if values:
            summary["dimensions"][family] = 100 * sum(values) / len(values)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--summary_json", type=str, default=None,
                        help="Optional path to save normalized summary JSON for paper builder")
    parser.add_argument("--variant_name", type=str, default=None,
                        help="Optional variant name written into the summary JSON")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--rollout", type=int, default=5)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--max_workers", type=int, default=10)
    args = parser.parse_args()

    # DashScope OpenAI-compatible endpoint
    client = OpenAI(
        api_key=args.api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model_name = "qwen2.5-vl-72b-instruct"

    # Load data
    json_data = []
    with open(args.input_json) as f:
        for line in f:
            if line.strip():
                json_data.append(json.loads(line))

    # Take first N samples
    json_data = json_data[:args.num_samples]
    print(f"Evaluating {len(json_data)} samples with {model_name}")
    print(f"Rollout: {args.rollout}, Count threshold: {args.count}, Workers: {args.max_workers}")

    all_results = []

    def worker(idx_data):
        idx, data = idx_data
        image_path = os.path.join(args.image_dir, f"{data['id']}.png")
        return process_item(idx, data, image_path, client, model_name, args.rollout, args.count)

    tasks = list(enumerate(json_data))

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(worker, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Evaluating"):
            result = future.result()
            if result:
                all_results.append(result)

    # Save results
    all_results.sort(key=lambda x: x['id'])
    with open(args.output_json, 'w') as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Calculate metrics
    if all_results:
        summary = build_paper_summary(all_results, variant_name=args.variant_name)

        print(f"\n=== Results ({len(all_results)} samples) ===")
        print(f"  avg_acc:     {summary['overall']['avg_acc']:.2f}")
        print(f"  basic_acc:   {summary['overall']['basic_acc']:.2f}")
        print(f"  spatial_acc: {summary['overall']['spatial_acc']:.2f}")

        if args.summary_json:
            summary_dir = os.path.dirname(args.summary_json)
            if summary_dir:
                os.makedirs(summary_dir, exist_ok=True)
            with open(args.summary_json, 'w') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"  summary_json: {args.summary_json}")

        return summary


if __name__ == "__main__":
    main()
