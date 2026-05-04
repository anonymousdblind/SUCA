from PIL import Image
import io
import numpy as np
import torch
from collections import defaultdict

def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew/500, meta

    return _fn

def aesthetic_score():
    from flow_grpo.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn

def clip_score(device):
    from flow_grpo.clip_scorer import ClipScorer

    scorer = ClipScorer(device=device)

    def _fn(images, prompts, metadata):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)/255.0
        scores = scorer(images, prompts)
        return scores, {}

    return _fn

def image_similarity_score(device):
    from flow_grpo.clip_scorer import ClipScorer

    scorer = ClipScorer(device=device).cuda()

    def _fn(images, ref_images):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)/255.0
        if not isinstance(ref_images, torch.Tensor):
            ref_images = [np.array(img) for img in ref_images]
            ref_images = np.array(ref_images)
            ref_images = ref_images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            ref_images = torch.tensor(ref_images, dtype=torch.uint8)/255.0
        scores = scorer.image_similarity(images, ref_images)
        return scores, {}

    return _fn

def pickscore_score(device):
    from flow_grpo.pickscore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

def imagereward_score(device):
    from flow_grpo.imagereward_scorer import ImageRewardScorer

    scorer = ImageRewardScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

def qwenvl_score(device):
    from flow_grpo.qwenvl import QwenVLScorer

    scorer = QwenVLScorer(dtype=torch.bfloat16, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

    
def ocr_score(device):
    from flow_grpo.ocr import OcrScorer

    scorer = OcrScorer()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn

def video_ocr_score(device):
    from flow_grpo.ocr import OcrScorer_video_or_image

    scorer = OcrScorer_video_or_image()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            if images.dim() == 4 and images.shape[1] == 3:
                images = images.permute(0, 2, 3, 1) 
            elif images.dim() == 5 and images.shape[2] == 3:
                images = images.permute(0, 1, 3, 4, 2)
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn

def deqa_score_remote(device):
    """Submits images to DeQA and computes a reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64
    url = "http://127.0.0.1:18086"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        all_scores = []
        for image_batch in images_batched:
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)
            response_data = pickle.loads(response.content)

            all_scores += response_data["outputs"]

        return all_scores, {}

    return _fn

def geneval_score(device):
    """Submits images to GenEval and computes a reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64
    url = "http://127.0.0.1:18085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadatas, only_strict):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadatas_batched = np.array_split(metadatas, np.ceil(len(metadatas) / batch_size))
        all_scores = []
        all_rewards = []
        all_strict_rewards = []
        all_group_strict_rewards = []
        all_group_rewards = []
        for image_batch, metadata_batched in zip(images_batched, metadatas_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "meta_datas": list(metadata_batched),
                "only_strict": only_strict,
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)
            response_data = pickle.loads(response.content)

            all_scores += response_data["scores"]
            all_rewards += response_data["rewards"]
            all_strict_rewards += response_data["strict_rewards"]
            all_group_strict_rewards.append(response_data["group_strict_rewards"])
            all_group_rewards.append(response_data["group_rewards"])
        all_group_strict_rewards_dict = defaultdict(list)
        all_group_rewards_dict = defaultdict(list)
        for current_dict in all_group_strict_rewards:
            for key, value in current_dict.items():
                all_group_strict_rewards_dict[key].extend(value)
        all_group_strict_rewards_dict = dict(all_group_strict_rewards_dict)

        for current_dict in all_group_rewards:
            for key, value in current_dict.items():
                all_group_rewards_dict[key].extend(value)
        all_group_rewards_dict = dict(all_group_rewards_dict)

        return all_scores, all_rewards, all_strict_rewards, all_group_rewards_dict, all_group_strict_rewards_dict

    return _fn

def unifiedreward_score_remote(device):
    """Submits images to DeQA and computes a reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64
    url = "http://10.82.120.15:18085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))

        all_scores = []
        for image_batch, prompt_batch in zip(images_batched, prompts_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "prompts": prompt_batch
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)
            print("response: ", response)
            print("response: ", response.content)
            response_data = pickle.loads(response.content)

            all_scores += response_data["outputs"]

        return all_scores, {}

    return _fn

def unifiedreward_score_sglang(device):
    import asyncio
    from openai import AsyncOpenAI
    import base64
    from io import BytesIO
    import re 

    def pil_image_to_base64(image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image_text}"
        return base64_qwen

    def _extract_scores(text_outputs):
        scores = []
        pattern = r"Final Score:\s*([1-5](?:\.\d+)?)"
        for text in text_outputs:
            match = re.search(pattern, text)
            if match:
                try:
                    scores.append(float(match.group(1)))
                except ValueError:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        return scores

    client = AsyncOpenAI(base_url="http://127.0.0.1:17140/v1", api_key="flowgrpo")
        
    async def evaluate_image(prompt, image):
        question = f"<image>\nYou are given a text caption and a generated image based on that caption. Your task is to evaluate this image based on two key criteria:\n1. Alignment with the Caption: Assess how well this image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nBased on the above criteria, assign a score from 1 to 5 after \'Final Score:\'.\nYour task is provided as follows:\nText Caption: [{prompt}]"
        images_base64 = pil_image_to_base64(image)
        response = await client.chat.completions.create(
            model="UnifiedReward-7b-v1.5",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": images_base64},
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    async def evaluate_batch_image(images, prompts):
        tasks = [evaluate_image(prompt, img) for prompt, img in zip(prompts, images)]
        results = await asyncio.gather(*tasks)
        return results

    def _fn(images, prompts, metadata):
        # 处理Tensor类型转换
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        
        # 转换为PIL Image并调整尺寸
        images = [Image.fromarray(image).resize((512, 512)) for image in images]

        # 执行异步批量评估
        text_outputs = asyncio.run(evaluate_batch_image(images, prompts))
        score = _extract_scores(text_outputs)
        score = [sc/5.0 for sc in score]
        return score, {}
    
    return _fn

def suca_vqa_score(device):
    """SUCA VQA reward: parse prompts into semantic units, score via Qwen3-VL FastAPI server."""
    import os as _os
    import sys as _sys
    import requests as _requests
    import base64 as _base64
    from io import BytesIO as _BytesIO
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "../.."))
    from suca.semantic_parser import SemanticUnitParser, filter_units_for_online_reward, validate_units

    import logging as _logging
    _reward_logger = _logging.getLogger("reward-diag")
    _reward_logger.setLevel(_logging.DEBUG)
    if not _reward_logger.handlers:
        _diag_log = "outputs/ablation/reward_diag.log"
        _os.makedirs(_os.path.dirname(_diag_log), exist_ok=True)
        _rh = _logging.FileHandler(_diag_log, mode="a")
        _rh.setFormatter(_logging.Formatter("%(asctime)s %(message)s"))
        _reward_logger.addHandler(_rh)

    parser = SemanticUnitParser(use_rules_only=True)
    # Auto-detect which reward server ports are actually alive
    _candidate_ports = [8100, 8101]
    ports = []
    for _p in _candidate_ports:
        try:
            _r = _requests.get(f"http://localhost:{_p}/health", timeout=3)
            if _r.status_code == 200:
                ports.append(_p)
        except Exception:
            pass
    if not ports:
        ports = [8100]  # fallback
    _reward_logger.info(f"Active reward server ports: {ports} (checked {_candidate_ports})")
    _idx = [0]
    _diag_count = [0]  # log first N samples in detail, then summary only

    def _score_image(image_np, prompt):
        """Score one image against its prompt's semantic units.
        Returns: (avg_score, per_unit_scores, unit_types)
        """
        units = parser.parse(prompt, online_mode=True)
        if not units:
            _reward_logger.warning(f"NO_UNITS prompt={prompt[:80]}")
            return 0.5, [], []

        vqa_qs = [u.vqa_question for u in units]
        unit_types = [u.unit_type.value for u in units]
        port = ports[_idx[0] % len(ports)]
        _idx[0] += 1

        pil = Image.fromarray(image_np)
        buf = _BytesIO()
        pil.save(buf, format="PNG")
        img_b64 = _base64.b64encode(buf.getvalue()).decode()

        try:
            resp = _requests.post(
                f"http://localhost:{port}/score_batch",
                json={"image_b64": img_b64, "questions": vqa_qs},
                timeout=120,
            )
            resp_data = resp.json()
            raw_scores = resp_data["scores"]
            # Use log-margin reward: log P(Yes) - log P(No)
            # Much better discriminability than raw probability in saturated region
            raw_margins = resp_data.get("margins", None)
            if raw_margins is None:
                # Fallback: compute margin from probability
                import math
                raw_margins = []
                for s in raw_scores:
                    s = max(min(s, 0.999), 0.001)
                    raw_margins.append(math.log(s) - math.log(1 - s))

            # Diagnostic logging
            is_fallback = all(abs(s - 0.5) < 1e-6 for s in raw_scores)
            if is_fallback or _diag_count[0] < 50:
                _diag_count[0] += 1
                tag = "FALLBACK" if is_fallback else "OK"
                _reward_logger.info(
                    f"[{tag}] port={port} prompt={prompt[:60]}... | "
                    f"n_units={len(vqa_qs)} raw={[f'{s:.3f}' for s in raw_scores]} "
                    f"margins={[f'{m:.2f}' for m in raw_margins]}"
                )
                for q, rs, mg in zip(vqa_qs, raw_scores, raw_margins):
                    _reward_logger.info(f"  Q: {q} | raw={rs:.4f} margin={mg:.2f}")

            # Use margins as per-unit scores for SUCA (high discriminability)
            # Average margin as the scalar reward
            avg = float(sum(raw_margins) / len(raw_margins)) if raw_margins else 0.0
            return avg, raw_margins, unit_types
        except Exception as e:
            _reward_logger.error(
                f"EXCEPTION port={port} prompt={prompt[:60]}... | "
                f"error={type(e).__name__}: {e} | n_qs={len(vqa_qs)}"
            )
            return 0.5, [0.5] * len(vqa_qs), unit_types

    def _fn(images, prompts, metadata, only_strict=False):
        if isinstance(images, torch.Tensor):
            images_np = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images_np = images_np.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        else:
            images_np = images

        # Parallel requests: send all images concurrently to reward servers
        from concurrent.futures import ThreadPoolExecutor as _TPE
        with _TPE(max_workers=min(len(images_np), 8)) as pool:
            futures = [
                pool.submit(_score_image, img, prompt)
                for img, prompt in zip(images_np, prompts)
            ]
            results = [f.result() for f in futures]

        scores = [r[0] for r in results]
        # Store per-unit breakdown in metadata for SUCA
        per_unit_data = {
            "per_unit_scores": [r[1] for r in results],
            "unit_types": [r[2] for r in results],
        }

        return scores, per_unit_data

    return _fn


def score_anchor_images(images_np, prompts, parser, ports, anchor_step,
                        unit_type_filter=None, confidence_gate=(0.3, 0.7)):
    """Score intermediate (anchor) images for sparse process reward.

    Args:
        images_np: numpy array (N, H, W, 3) uint8
        prompts: list of prompt strings
        parser: SemanticUnitParser instance
        ports: list of reward server ports
        anchor_step: which denoising step this is (for logging)
        unit_type_filter: only score these unit types, e.g. ["entity","count"]. None = all.
        confidence_gate: (low, high) — raw probs in this range are uninformative

    Returns:
        list of dicts, one per sample: {unit_idx: margin_score} (None for gated-out units)
    """
    import requests as _requests
    import base64 as _base64
    from io import BytesIO as _BytesIO
    import math
    import logging as _logging
    _logger = _logging.getLogger("process-reward")

    results = []
    for i, (img_np, prompt) in enumerate(zip(images_np, prompts)):
        try:
            units = parser.parse(prompt, online_mode=True)
            if not units:
                results.append({})
                continue

            # Filter units by type if specified
            filtered_indices = []
            filtered_questions = []
            for k, u in enumerate(units):
                if unit_type_filter is None or u.unit_type.value in unit_type_filter:
                    filtered_indices.append(k)
                    filtered_questions.append(u.vqa_question)

            if not filtered_questions:
                results.append({})
                continue

            # Encode image
            pil = Image.fromarray(img_np)
            buf = _BytesIO()
            pil.save(buf, format="PNG")
            img_b64 = _base64.b64encode(buf.getvalue()).decode()

            # Score via reward server
            port = ports[i % len(ports)]
            resp = _requests.post(
                f"http://localhost:{port}/score_batch",
                json={"image_b64": img_b64, "questions": filtered_questions},
                timeout=120,
            )
            resp_data = resp.json()
            raw_scores = resp_data["scores"]
            raw_margins = resp_data.get("margins", None)
            if raw_margins is None:
                raw_margins = []
                for s in raw_scores:
                    s = max(min(s, 0.999), 0.001)
                    raw_margins.append(math.log(s) - math.log(1 - s))

            # Apply confidence gating
            unit_scores = {}
            for j, (k, margin, raw_p) in enumerate(zip(filtered_indices, raw_margins, raw_scores)):
                if confidence_gate[0] < raw_p < confidence_gate[1]:
                    unit_scores[k] = None  # uninformative
                else:
                    unit_scores[k] = margin

            results.append(unit_scores)

        except Exception as e:
            _logger.warning(f"[anchor t={anchor_step}] sample {i} error: {e}")
            results.append({})

    return results


def multi_score(device, score_dict):
    score_functions = {
        "deqa": deqa_score_remote,
        "ocr": ocr_score,
        "video_ocr": video_ocr_score,
        "imagereward": imagereward_score,
        "pickscore": pickscore_score,
        "qwenvl": qwenvl_score,
        "aesthetic": aesthetic_score,
        "jpeg_compressibility": jpeg_compressibility,
        "unifiedreward": unifiedreward_score_sglang,
        "geneval": geneval_score,
        "suca_vqa": suca_vqa_score,
        "clipscore": clip_score,
        "image_similarity": image_similarity_score,
    }
    score_fns={}
    for score_name, weight in score_dict.items():
        score_fns[score_name] = score_functions[score_name](device) if 'device' in score_functions[score_name].__code__.co_varnames else score_functions[score_name]()

    # only_strict is only for geneval. During training, only the strict reward is needed, and non-strict rewards don't need to be computed, reducing reward calculation time.
    def _fn(images, prompts, metadata, ref_images=None, only_strict=True):
        total_scores = []
        score_details = {}
        
        for score_name, weight in score_dict.items():
            if score_name == "geneval":
                scores, rewards, strict_rewards, group_rewards, group_strict_rewards = score_fns[score_name](images, prompts, metadata, only_strict)
                score_details['accuracy'] = rewards
                score_details['strict_accuracy'] = strict_rewards
                for key, value in group_strict_rewards.items():
                    score_details[f'{key}_strict_accuracy'] = value
                for key, value in group_rewards.items():
                    score_details[f'{key}_accuracy'] = value
            elif score_name == "image_similarity":
                scores, rewards = score_fns[score_name](images, ref_images)
            else:
                scores, rewards = score_fns[score_name](images, prompts, metadata)
            score_details[score_name] = scores
            weighted_scores = [weight * score for score in scores]

            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]

            # Pass through per-unit data from suca_vqa for SUCA credit assignment
            if score_name == "suca_vqa" and isinstance(rewards, dict) and "per_unit_scores" in rewards:
                score_details["_per_unit_metadata"] = rewards

        score_details['avg'] = total_scores
        # Extract per-unit metadata if available
        metadata_out = {}
        if "_per_unit_metadata" in score_details:
            metadata_out = score_details.pop("_per_unit_metadata")
        return score_details, metadata_out

    return _fn

def main():
    import torchvision.transforms as transforms

    image_paths = [
        "nasa.jpg",
    ]

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
    ])

    images = torch.stack([transform(Image.open(image_path).convert('RGB')) for image_path in image_paths])
    prompts=[
        'A astronaut’s glove floating in zero-g with "NASA 2049" on the wrist',
    ]
    metadata = {}  # Example metadata
    score_dict = {
        "unifiedreward": 1.0
    }
    # Initialize the multi_score function with a device and score_dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scoring_fn = multi_score(device, score_dict)
    # Get the scores
    scores, _ = scoring_fn(images, prompts, metadata)
    # Print the scores
    print("Scores:", scores)


if __name__ == "__main__":
    main()
