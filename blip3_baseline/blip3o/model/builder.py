# blip3o/model/builder.py  (fixed)

import os
import warnings
import shutil
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
)

from blip3o.model import *  # keeps your repo's model class imports
from blip3o.constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from blip3o.train.train import smart_tokenizer_and_embedding_resize


def _post_tokenizer_expansion(tokenizer, model):
    """
    Add special image tokens if the config expects them and resize embeddings.
    """
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)

    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    model.resize_token_embeddings(len(tokenizer),pad_to_multiple_of=8)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    return context_len


def load_pretrained_model(model_path, tokenizer_path, model_name):
    """
    Main loader used by inference.py
    Loads the BLIP3/Qwen model in 4-bit (nf4) with device_map='auto' so it fits on 16GB GPUs.
    Returns (tokenizer, model, context_len)
    """
    # --- tokenizer ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=True, trust_remote_code=True
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, trust_remote_code=True
        )

    # --- 4-bit quantization config ---
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # NOTE: blip3oQwenForInferenceLM must be importable from blip3o.model (as in repo)
    # If your repo uses a different class name, keep it consistent.
    model = blip3oQwenForInferenceLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        quantization_config=bnb_cfg,
        device_map="auto",          # let HF place across available devices
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    context_len = _post_tokenizer_expansion(tokenizer, model)
    return tokenizer, model, context_len


def load_pretrained_model_lmms_eval(
    model_path,
    load_8bit: bool = False,
    load_4bit: bool = True,
    device_map: str = "auto",
    device: str = "cuda",
    use_flash_attn: bool = False,
    **kwargs,
):
    """
    Secondary loader for eval scripts. Defaults to 4-bit as well.
    Returns (tokenizer, model, context_len)
    """
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs["device_map"] = {"": device}

    if load_8bit:
        kwargs["load_in_8bit"] = True
        kwargs.pop("quantization_config", None)
        kwargs.pop("torch_dtype", None)
    elif load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        kwargs.pop("torch_dtype", None)
    else:
        kwargs["torch_dtype"] = torch.float16
        kwargs.pop("quantization_config", None)

    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, trust_remote_code=True
    )

    model = blip3oQwenForInferenceLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        device_map=kwargs.get("device_map", "auto"),
        quantization_config=kwargs.get("quantization_config", None),
        torch_dtype=kwargs.get("torch_dtype", None),
        attn_implementation=kwargs.get("attn_implementation", None),
        trust_remote_code=True,
    )

    context_len = _post_tokenizer_expansion(tokenizer, model)
    return tokenizer, model, context_len

