#!/usr/bin/env python3
import argparse
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


def load_vlm(model_id):
    """
    Return (tokenizer, model, processor) for a Vision-Language Model (VLM).
    """
    from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM
    import torch

    use_vl = any(x in model_id for x in ["VL", "-VL-", "Qwen2.5-VL"])
    if use_vl:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        # Let the repo provide the correct class via trust_remote_code
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16
        )
        tokenizer = getattr(processor, "tokenizer", AutoTokenizer.from_pretrained(model_id, trust_remote_code=True))
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16
        )
        processor = None
    return tokenizer, model, processor

def run_vlm(model, processor, prompt: str, image_path: Optional[str], max_new_tokens: int, temperature: float) -> str:
    # Text-only path (e.g., Qwen/Qwen2.5-7B-Instruct)
    if processor is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model.config._name_or_path, trust_remote_code=True)
        # For text-only models, content must be a STRING, not a list of blocks
        messages = [{"role": "user", "content": prompt}]
        try:
            chat_prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        except Exception:
            # Fallback if template unavailable
            chat_prompt = prompt
        # Ensure we pass a string to the tokenizer
        if not isinstance(chat_prompt, str):
            chat_prompt = str(chat_prompt)
        inputs = tok(chat_prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=True)
        gen_ids = out_ids[:, inputs['input_ids'].shape[1]:]
        return tok.batch_decode(gen_ids, skip_special_tokens=True)[0]

    # VL path (processor present)
    if image_path:
        messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]}]
    else:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    chat_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[chat_prompt], images=image_inputs, videos=video_inputs, return_tensors="pt", padding=True)
    inputs, device = to_device(inputs)
    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=True)
    gen_ids = out_ids[:, inputs['input_ids'].shape[1]:]
    text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    return text


def load_t2i_pipeline(model_id: str, device: str):
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32, safety_checker=None, requires_safety_checker=False)
    pipe = pipe.to(device)
    return pipe


def load_img2img_pipeline(model_id: str, device: str):
    from diffusers import StableDiffusionImg2ImgPipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32, safety_checker=None, requires_safety_checker=False)
    pipe = pipe.to(device)
    return pipe


def load_inpaint_pipeline(model_id: str, device: str):
    from diffusers import StableDiffusionInpaintPipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32, safety_checker=None, requires_safety_checker=False)
    pipe = pipe.to(device)
    return pipe


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, help="Mode: t2t, i2t, t2i, or i2i")
    parser.add_argument("--vlm", type=str, default=None, help="Vision-language model name")
    parser.add_argument("--diffusion", type=str, default="runwayml/stable-diffusion-v1-5", help="Diffusion model for t2i or i2i modes")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--mask", type=str, default=None, help="Optional mask image for inpainting")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for the model")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max new tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--steps", type=int, default=30, help="Number of diffusion steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale for diffusion")
    parser.add_argument("--strength", type=float, default=0.8, help="How much to transform the input image (0.0–1.0)")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    parser.add_argument("--use_lora", type=int, default=0, help="Use a PEFT LoRA adapter")
    parser.add_argument("--peft_path", type=str, default=None, help="Path to LoRA adapter folder")
    args = parser.parse_args()
    if not hasattr(args, 'output_dir') or not args.output_dir:
        args.output_dir = 'results'
    print('[DEBUG] mode=', args.mode, 'vlm=', args.vlm)

    set_global_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ensure_dir(args.output_dir)

    if args.mode in ('t2t', 'i2t'):
        tokenizer, vlm_model, processor = load_vlm(args.vlm)
        if args.use_lora and args.peft_path:
            from peft import PeftModel
            vlm_model = PeftModel.from_pretrained(vlm_model, args.peft_path)
            try:
                vlm_model = vlm_model.merge_and_unload()
            except Exception as e:
                print("[WARN] Could not merge LoRA:", e)
        if args.mode == 't2t':
            result = run_vlm(
                vlm_model, processor, args.prompt,
                image_path=None,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            print('>>>', result)
            with open(os.path.join(args.output_dir, f't2t_{timestamp()}.txt'), 'w') as f:
                f.write(result.strip() + '\n')
        else:
            if not args.image or not os.path.exists(args.image):
                raise FileNotFoundError('--image is required for i2t and must exist')
            result = run_vlm(
                vlm_model, processor, args.prompt,
                image_path=args.image,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            print('>>>', result)

        return

    if args.mode == "t2i":
        pipe = load_t2i_pipeline(args.diffusion, device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device=="cuda")):
            image = pipe(prompt=args.prompt, num_inference_steps=args.steps, guidance_scale=args.guidance).images[0]
        out_path = os.path.join(args.output_dir, f"t2i_{timestamp()}.png")
        image.save(out_path)
        print(f"Saved: {out_path}")
        return

    if args.mode == "i2i":
        if not args.image or not os.path.exists(args.image):
            raise FileNotFoundError("--image is required for i2i and must exist")
        init_image = Image.open(args.image).convert("RGB")
        if args.mask:
            if not os.path.exists(args.mask):
                raise FileNotFoundError(f"Mask not found: {args.mask}")
            mask_image = Image.open(args.mask).convert("RGB")
            pipe = load_inpaint_pipeline(args.diffusion, device)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device=="cuda")):
                image = pipe(prompt=args.prompt, image=init_image, mask_image=mask_image, num_inference_steps=args.steps, guidance_scale=args.guidance).images[0]
            base = os.path.splitext(os.path.basename(args.image))[0]
            out_path = os.path.join(args.output_dir, f"{base}_inpaint_{timestamp()}.png")
            image.save(out_path)
            print(f"Saved: {out_path}")
        else:
            pipe = load_img2img_pipeline(args.diffusion, device)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device=="cuda")):
                image = pipe(prompt=args.prompt, image=init_image, strength=args.strength, guidance_scale=args.guidance, num_inference_steps=args.steps).images[0]
            base = os.path.splitext(os.path.basename(args.image))[0]
            out_path = os.path.join(args.output_dir, f"{base}_img2img_{timestamp()}.png")
            image.save(out_path)
            print(f"Saved: {out_path}")
        return


if __name__ == "__main__":
    main()