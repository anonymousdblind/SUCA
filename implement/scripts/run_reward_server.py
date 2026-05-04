"""
Simple VQA reward server using Flask + Qwen3-VL.
Runs as a standalone process on GPU 6 (or 6,7).
The trainer sends HTTP requests instead of loading VLM in-process.
"""
import argparse
import base64
import json
import logging
import os
import sys
from io import BytesIO

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reward-server")

app = FastAPI()
model = None
processor = None
device = None


class ScoreRequest(BaseModel):
    image_b64: str
    questions: list


def load_model(model_path, gpu_id):
    global model, processor, device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"

    import transformers
    from transformers import AutoProcessor
    # Auto-detect model class based on config
    import json as _json
    with open(os.path.join(model_path, "config.json")) as _f:
        _cfg = _json.load(_f)
    arch = _cfg.get("architectures", [""])[0]
    logger.info(f"Loading {arch} from {model_path} on GPU {gpu_id}...")

    auto_image_text_cls = getattr(transformers, "AutoModelForImageTextToText", None)
    auto_vision2seq_cls = getattr(transformers, "AutoModelForVision2Seq", None)
    auto_model_cls = getattr(transformers, "AutoModel", None)

    model_cls = getattr(transformers, arch, None)
    if model_cls is None and "Qwen" in arch:
        if auto_image_text_cls is not None:
            logger.warning(
                "Transformers does not expose %s; falling back to AutoModelForImageTextToText",
                arch,
            )
            model_cls = auto_image_text_cls
        elif auto_vision2seq_cls is not None:
            logger.warning(
                "Transformers does not expose %s; falling back to AutoModelForVision2Seq",
                arch,
            )
            model_cls = auto_vision2seq_cls
        elif auto_model_cls is not None:
            logger.warning(
                "Transformers does not expose %s; falling back to AutoModel",
                arch,
            )
            model_cls = auto_model_cls
    elif model_cls is None:
        if auto_vision2seq_cls is not None:
            logger.warning(
                "Unknown vision-language architecture %s; falling back to AutoModelForVision2Seq",
                arch,
            )
            model_cls = auto_vision2seq_cls
        elif auto_model_cls is not None:
            logger.warning(
                "Unknown vision-language architecture %s; falling back to AutoModel",
                arch,
            )
            model_cls = auto_model_cls

    if model_cls is None:
        raise ImportError(
            "No compatible model loader found in transformers. "
            "Upgrade transformers to a version that supports this VLM architecture, e.g. 4.57.6."
        )

    if auto_image_text_cls is not None and model_cls is auto_image_text_cls:
        logger.info("Using AutoModelForImageTextToText fallback for %s", arch)
    elif auto_vision2seq_cls is not None and model_cls is auto_vision2seq_cls:
        logger.info("Using AutoModelForVision2Seq fallback for %s", arch)
    elif auto_model_cls is not None and model_cls is auto_model_cls:
        logger.info("Using AutoModel fallback for %s", arch)
    else:
        logger.info("Using resolved model class %s", getattr(model_cls, "__name__", str(model_cls)))

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    # Decoder-only models require left-padding for correct batched generation
    processor.tokenizer.padding_side = "left"
    model = model_cls.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()
    logger.info("Model loaded!")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score_batch")
def score_batch(req: ScoreRequest):
    """Score a batch of VQA questions for a single image using batched inference."""
    data = req.dict()
    img_b64 = data["image_b64"]
    questions = data["questions"]

    img_bytes = base64.b64decode(img_b64)
    image = Image.open(BytesIO(img_bytes)).convert("RGB")

    if not questions:
        return {"scores": []}

    # Cache yes/no token ids (computed once)
    global _yes_id, _no_id
    if not hasattr(score_batch, '_yes_id'):
        score_batch._yes_id = processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
        score_batch._no_id = processor.tokenizer.encode("No", add_special_tokens=False)[0]
    yes_id = score_batch._yes_id
    no_id = score_batch._no_id

    # --- Batch inference: process all questions in one forward pass ---
    # Build messages for each question (same image, different text)
    all_texts = []
    all_images = []
    for q in questions:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"{q} Answer in one word."},
            ],
        }]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,  # Qwen3-VL: disable thinking for Yes/No classification
        )
        all_texts.append(text)
        all_images.append(image)

    # Batch process — one forward pass for all questions
    try:
        inputs = processor(
            text=all_texts, images=all_images, padding=True, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            # Single batched forward pass (much faster than N sequential passes)
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Extract per-question scores from batched output
        first_logits = outputs.scores[0]  # [batch_size, vocab_size]
        probs = torch.softmax(first_logits, dim=-1)

        scores = []
        margins = []
        for i in range(len(questions)):
            yes_prob = probs[i, yes_id].item()
            no_prob = probs[i, no_id].item()
            if yes_prob + no_prob > 0:
                norm_yes = yes_prob / (yes_prob + no_prob)
                scores.append(float(norm_yes))
                # Log margin: log P(Yes) - log P(No), clamped for numerical stability
                import math
                lp_yes = math.log(max(yes_prob, 1e-8))
                lp_no = math.log(max(no_prob, 1e-8))
                margins.append(float(lp_yes - lp_no))
            else:
                gen_tokens = outputs.sequences[i][inputs.input_ids.shape[1]:]
                gen_text = processor.decode(gen_tokens, skip_special_tokens=True).strip().lower()
                scores.append(0.9 if "yes" in gen_text else 0.1 if "no" in gen_text else 0.5)
                margins.append(2.0 if "yes" in gen_text else -2.0 if "no" in gen_text else 0.0)

    except torch.cuda.OutOfMemoryError:
        # Fallback: if batch too large, process in mini-batches of 4
        logger.warning(f"OOM with batch={len(questions)}, falling back to mini-batch")
        torch.cuda.empty_cache()
        scores = []
        margins = []
        for i in range(0, len(questions), 4):
            mini_texts = all_texts[i:i+4]
            mini_images = all_images[i:i+4]
            try:
                inputs = processor(
                    text=mini_texts, images=mini_images, padding=True, return_tensors="pt"
                ).to(device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, max_new_tokens=5, do_sample=False,
                        return_dict_in_generate=True, output_scores=True,
                    )
                first_logits = outputs.scores[0]
                probs = torch.softmax(first_logits, dim=-1)
                for j in range(len(mini_texts)):
                    yp = probs[j, yes_id].item()
                    np_ = probs[j, no_id].item()
                    if yp + np_ > 0:
                        scores.append(float(yp / (yp + np_)))
                        margins.append(float(math.log(max(yp,1e-8)) - math.log(max(np_,1e-8))))
                    else:
                        scores.append(0.5)
                        margins.append(0.0)
            except Exception as e:
                logger.error(f"Mini-batch error: {e}")
                scores.extend([0.5] * len(mini_texts))
                margins.extend([0.0] * len(mini_texts))

    except Exception as e:
        logger.warning(f"Batch VQA error ({len(questions)} qs): {e}, falling back to sequential")
        scores = []
        margins = []
        for i in range(len(questions)):
            try:
                inp = processor(
                    text=[all_texts[i]], images=[all_images[i]], return_tensors="pt"
                ).to(device)
                with torch.no_grad():
                    out = model.generate(
                        **inp, max_new_tokens=5, do_sample=False,
                        return_dict_in_generate=True, output_scores=True,
                    )
                logits_i = out.scores[0][0]
                probs_i = torch.softmax(logits_i, dim=-1)
                yp = probs_i[yes_id].item()
                np_ = probs_i[no_id].item()
                if yp + np_ > 0:
                    scores.append(float(yp / (yp + np_)))
                    margins.append(float(math.log(max(yp,1e-8)) - math.log(max(np_,1e-8))))
                else:
                    scores.append(0.5)
                    margins.append(0.0)
            except Exception:
                scores.append(0.5)
                margins.append(0.0)

    return {"scores": scores, "margins": margins}


# ============================================
# GenEval-compatible endpoint (port 18085 protocol)
# Receives pickle(images + metadata), returns strict binary scores
# ============================================
from fastapi import Request
import pickle
import numpy as np

@app.post("/")
async def geneval_endpoint(request: Request):
    """Compatible with Flow-GRPO's geneval reward server protocol."""
    import io
    body = await request.body()
    data = pickle.loads(body)

    images_bytes = data["images"]  # list of JPEG bytes
    meta_datas = data["meta_datas"]  # list of metadata dicts
    only_strict = data.get("only_strict", True)

    scores = []
    rewards = []
    strict_rewards = []
    group_strict_rewards = {}
    group_rewards = {}

    for img_bytes, meta in zip(images_bytes, meta_datas):
        try:
            pil_img = Image.open(io.BytesIO(img_bytes))
            # Extract VQA questions from metadata
            vqa_list = meta.get("vqa_list", [])
            if not vqa_list:
                scores.append(0.5)
                rewards.append(0.5)
                strict_rewards.append(0.0)
                continue

            questions = [q for q, a in vqa_list]
            # Score using our VLM
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=85)
            img_b64 = base64.b64encode(buf.getvalue()).decode()

            resp = score_batch(ScoreRequest(image_b64=img_b64, questions=questions))
            soft_scores = resp["scores"]

            # Binary strict: ALL units must pass (>0.5) for reward=1
            binary = [1.0 if s > 0.5 else 0.0 for s in soft_scores]
            strict = 1.0 if all(b == 1.0 for b in binary) else 0.0
            avg = sum(binary) / len(binary)

            scores.append(soft_scores)
            rewards.append(avg)
            strict_rewards.append(strict)

            # Group by skill type if available
            skills = meta.get("skills", [])
            for i, skill in enumerate(skills):
                if i < len(binary):
                    group_strict_rewards.setdefault(skill, []).append(binary[i])
                    group_rewards.setdefault(skill, []).append(soft_scores[i] if i < len(soft_scores) else 0.5)

        except Exception as e:
            logger.error(f"GenEval scoring error: {e}")
            scores.append([0.5])
            rewards.append(0.5)
            strict_rewards.append(0.0)

    result = {
        "scores": scores,
        "rewards": rewards,
        "strict_rewards": strict_rewards,
        "group_strict_rewards": group_strict_rewards,
        "group_rewards": group_rewards,
    }
    from fastapi.responses import Response
    return Response(content=pickle.dumps(result), media_type="application/octet-stream")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()

    load_model(args.model, args.gpu)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)
