#!/usr/bin/env python3
import argparse
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS_DIR = os.path.join(BASE_DIR, "results")

import os
import time
from typing import Optional
import torch
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info


def set_global_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def to_device(batch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    moved = {}
    for k, v in batch.items():
        moved[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return moved, device


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def mode_subfolder(mode: str) -> str:
    if mode == "t2t":
        return "texttotext"
    if mode == "i2t":
        return "imagetotext"
    if mode == "t2i":
        return "texttoimage"
    if mode == "i2i":
        return "imagetoimage"
    return "other"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


###############################
# FIXED QWEN MODEL LOADER
###############################
def load_vlm(model_id):
    """
    Return (tokenizer, model, processor) for Qwen2.5-VL OR text-only LLM.
    """
    import torch
    from transformers import (
        AutoProcessor,
        AutoTokenizer,
        AutoModelForCausalLM,
        AutoConfig,
    )

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model_type = getattr(config, "model_type", None)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # ---- Qwen2.5-VL multimodal ----
    if model_type == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        model.to(device)

        tokenizer = getattr(
            processor,
            "tokenizer",
            AutoTokenizer.from_pretrained(model_id, trust_remote_code=True),
        )
        return tokenizer, model, processor

    # ---- Text-only fallback ----
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    processor = None
    return tokenizer, model, processor


######################################
# RUN QWEN MODEL
######################################
def run_vlm(model, processor, prompt: str, image_path: Optional[str], max_new_tokens: int, temperature: float) -> str:

    # --- Text-only ---
    if processor is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model.config._name_or_path, trust_remote_code=True)

        messages = [{"role": "user", "content": prompt}]
        try:
            chat_prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        except Exception:
            chat_prompt = prompt

        if not isinstance(chat_prompt, str):
            chat_prompt = str(chat_prompt)

        inputs = tok(chat_prompt, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
            )
        gen_ids = out_ids[:, inputs['input_ids'].shape[1]:]
        return tok.batch_decode(gen_ids, skip_special_tokens=True)[0]

    # --- Vision-language Qwen2.5-VL ---
    if image_path:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt}
        ]}]
    else:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    chat_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[chat_prompt],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True
    )
    inputs, device = to_device(inputs)

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
        )

    gen_ids = out_ids[:, inputs['input_ids'].shape[1]:]
    text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    return text


#########################################
# DIFFUSION PIPELINES
#########################################
def load_t2i_pipeline(model_id: str, device: str):
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    return pipe.to(device)


def load_img2img_pipeline(model_id: str, device: str):
    from diffusers import StableDiffusionImg2ImgPipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    return pipe.to(device)


def load_inpaint_pipeline(model_id: str, device: str):
    from diffusers import StableDiffusionInpaintPipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    return pipe.to(device)


#########################################
# MAIN
#########################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--vlm", type=str, default=None)
    parser.add_argument("--diffusion", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--mask", type=str, default=None)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--strength", type=float, default=0.8)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("[DEBUG] mode=", args.mode, "vlm=", args.vlm)

    set_global_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create base results folder
    ensure_dir(args.output_dir)

    ############################
    # TEXT → TEXT
    ############################
    if args.mode == "t2t":
        tokenizer, vlm_model, processor = load_vlm(args.vlm)
        result = run_vlm(vlm_model, processor, args.prompt, None, args.max_new_tokens, args.temperature)

        sub = mode_subfolder("t2t")
        out_dir = os.path.join(args.output_dir, sub)
        ensure_dir(out_dir)

        out_file = os.path.join(out_dir, f"t2t_{timestamp()}.txt")
        with open(out_file, "w") as f:
            f.write(result + "\n")

        print(">>>", result)
        return

    ############################
    # IMAGE → TEXT
    ############################
    if args.mode == "i2t":
        if not args.image or not os.path.exists(args.image):
            raise FileNotFoundError("--image is required")

        tokenizer, vlm_model, processor = load_vlm(args.vlm)
        result = run_vlm(vlm_model, processor, args.prompt, args.image, args.max_new_tokens, args.temperature)

        sub = mode_subfolder("i2t")
        out_dir = os.path.join(args.output_dir, sub)
        ensure_dir(out_dir)

        out_file = os.path.join(out_dir, f"i2t_{timestamp()}.txt")
        with open(out_file, "w") as f:
            f.write(result + "\n")

        print(">>>", result)
        return

    ############################
    # TEXT → IMAGE
    ############################
    if args.mode == "t2i":
        pipe = load_t2i_pipeline(args.diffusion, device)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=(device=="cuda")):
            image = pipe(
                prompt=args.prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance
            ).images[0]

        sub = mode_subfolder("t2i")
        out_dir = os.path.join(args.output_dir, sub)
        ensure_dir(out_dir)

        out_path = os.path.join(out_dir, f"t2i_{timestamp()}.png")
        image.save(out_path)
        print("Saved:", out_path)
        return

    ############################
    # IMAGE → IMAGE
    ############################
    if args.mode == "i2i":
        if not args.image or not os.path.exists(args.image):
            raise FileNotFoundError("--image is required")

        init_image = Image.open(args.image).convert("RGB")

        sub = mode_subfolder("i2i")
        out_dir = os.path.join(args.output_dir, sub)
        ensure_dir(out_dir)

        if args.mask:
            pipe = load_inpaint_pipeline(args.diffusion, device)
            mask_image = Image.open(args.mask).convert("RGB")

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=(device=="cuda")):
                image = pipe(
                    prompt=args.prompt,
                    image=init_image,
                    mask_image=mask_image,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance
                ).images[0]

            out_file = os.path.join(out_dir, f"inpaint_{timestamp()}.png")
            image.save(out_file)
            print("Saved:", out_file)

        else:
            pipe = load_img2img_pipeline(args.diffusion, device)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=(device=="cuda")):
                image = pipe(
                    prompt=args.prompt,
                    image=init_image,
                    strength=args.strength,
                    guidance_scale=args.guidance,
                    num_inference_steps=args.steps
                ).images[0]

            base = os.path.splitext(os.path.basename(args.image))[0]
            out_file = os.path.join(out_dir, f"{base}_{timestamp()}.png")
            image.save(out_file)
            print("Saved:", out_file)

        return


if __name__ == "__main__":
    main()
