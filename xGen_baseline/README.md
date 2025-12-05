# xGen-MM-Phi3 Frame Description Pipeline

This module contains scripts for **fine-tuning** and **running inference** on the multimodal model
**`Salesforce/xgen-mm-phi3-mini-instruct-singleimg-r-v1.5`**, adapted to describe frames extracted from farming-related videos.

The pipeline trains the model to generate **one short, clear, detailed sentence** per frame, using both the image and a text prompt.

---

> **Note:** Some folders still have `"blip3"` in their names for legacy reasons,
> but the actual model is **xGen-MM-Phi3**.

---

## Dataset Format (`blip3_frames.jsonl`)

Each line contains a single training example:

```json
{
  "image_path": "/path/to/frame.jpg",
  "input_text": "Question: Describe this frame in one short, clear, detailed sentence that helps understand the farming or product.",
  "answer_text": "A person uses soil testing equipment to examine nutrient levels.",
  "video_index": "-0BNev8b8CM",
  "task_index": 0,
  "subtask_index": 1,
  "frame_index": 2
}
```

### Required fields

- **`image_path`** – absolute path to the frame image
- **`input_text`** – prompt shown to the model (contains the question)
- **`answer_text`** – gold target sentence

### Optional metadata

- `video_index`
- `task_index`
- `subtask_index`
- `frame_index`

These are not used for training directly, but are used to group frames and sort them in the per-video reports.

---

## Model

### Model Name

```text
Salesforce/xgen-mm-phi3-mini-instruct-singleimg-r-v1.5
```

### Characteristics

- Multimodal (image + text) Phi-3–based architecture
- Vision input is represented as a 6D tensor:
  `(B, T_img, F, V, C, H, W)`
- Uses an internal **vision tokenizer** that expects a corresponding attention mask

The training and inference scripts:

- Patch the vision tokenizer so the **vision attention mask becomes optional**.
- Convert the image tensor from standard shapes (`[B, C, H, W]`, etc.) into the correct 6D layout expected by the model.
- Use the same preprocessing in both training and inference for consistency.

---

## Training

From a GPU-enabled environment:

```bash
conda activate blip3
cd VideoProcessing/scripts

python train.py
```

The training script will:

1. Load the JSONL dataset from:

   ```text
   results/training_data/blip3_frames.jsonl
   ```

2. For each example:
   - Load the image from `image_path`.
   - Build a prompt of the form:

     ```text
     You are an assistant that explains farming videos.

     {input_text}

     Answer:
     ```

   - Tokenize the full prompt + answer.
   - Mask **only the prompt tokens** in the labels with `-100`, so the model is trained to predict only the answer.

3. Run a simple fine-tuning loop using SGD.

4. Save the fine-tuned model, tokenizer, and image processor to:

   ```text
   results/blip3_full_finetune/
   ```

### Key Hyperparameters

These are defined inside `train_blip3_frames.py`:

- `MAX_LENGTH = 256`
- `BATCH_SIZE = 1`
- `NUM_EPOCHS = 1` (increase for real training)
- `LR = 2e-5`
- `MAX_SAMPLES = 20` (for quick tests; set to `None` to use the full dataset)

---

## Inference: Per-Video Reports

To run inference and generate human-readable reports grouped by video:

```bash
conda activate blip3
cd VideoProcessing/scripts

python xgen.py
```

This script will:

1. Load the fine-tuned model from:

   ```text
   results/blip3_full_finetune/
   ```

2. Load the same JSONL dataset, group entries by `video_index`, and sort frames by
   `(task_index, subtask_index, frame_index)`.

3. For each frame:
   - Load the image, process it with the image processor, and build `vision_x` with the correct 6D shape.
   - Reconstruct the same training-style prompt:

     ```text
     You are an assistant that explains farming videos.

     {input_text}

     Answer:
     ```

   - Generate a continuation with greedy decoding.
   - Strip everything before `"Answer:"` and keep only the model’s answer.

4. Write one `.txt` report file per video to:

   ```text
   results/xgen/{video_index}.txt
   ```

### Example Output Snippet

```text
==================== VIDEO -0BNev8b8CM ====================

------------------------------------------------------------
Task 0 | Subtask 0 | Frame: task00_sub00_f00.jpg

Task / Question:
Describe this frame in one short, clear, detailed sentence that helps understand the farming or product.

Ground Truth:
The video frame shows a financial market display with a text overlay indicating the growth of the Soil Fertility Testing Market.

Model Prediction:
The soil fertility testing market is expected to grow significantly by 2034, with a predicted size of $2.4 billion.
------------------------------------------------------------
```

The `Task / Question` line prefers a higher-level task description if such a field is present in the JSON. Otherwise, it falls back to extracting the question from `input_text` (after `"Question:"`).

---

## Goal

The goal of this pipeline is to fine-tune **xGen-MM-Phi3** so that, given a single frame from a farming-related video and a short prompt:

- It produces **one concise sentence**,
- grounded in the **visual content**,
- that helps explain the farming context, product, or message in the frame.

This is useful for:
- Video understanding
- Dataset creation (captions/annotations)
- Downstream tasks like retrieval or explanation for agricultural videos

---

## Environment & Caching (HPC-friendly)

On an HPC system (e.g., Bridges2), it’s recommended to redirect HuggingFace caches to a project directory:

```bash
export HF_HOME=/ocean/projects/PROJECT_ID/USER/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME
export XDG_CACHE_HOME=$HF_HOME
mkdir -p "$HF_HOME"
```

Then activate your environment and verify GPU access:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate blip3

nvidia-smi
```

Now you can run the training and inference scripts without repeatedly downloading models.

---
# Frame Captioning with InstructBLIP

This script generates **detailed captions** for video frames using the **InstructBLIP (FLAN-T5-XL)** model.  
It reads a JSONL dataset of frames, processes each frame, and outputs **one text report per video**.

## What the Script Does (instrutblip.py)
1. Loads the InstructBLIP vision-language model.
2. Reads a JSONL file containing frame metadata and image paths.
3. Groups frames by `video_index`.
4. For each frame:
   - Opens the image
   - Extracts the question from the input text
   - Generates a 2–4 sentence detailed caption
5. Saves a `.txt` report for each video showing:
   - Input question  
   - Ground-truth answer  
   - Model prediction  

## Input Format (JSONL)
Each line should contain fields like:

{
  "video_index": 0,
  "image_path": "/path/to/frame.png",
  "input_text": "Question: What is happening in this frame?",
  "answer_text": "A farmer checks plants inside a greenhouse."
}

## Output
The script writes one text file per video in the output directory:

video_0.txt  
video_1.txt  
...

Each file contains neatly formatted blocks comparing ground truth and model output.

## Installation
pip install torch datasets transformers pillow

## Run
python instructblip.py


## Possible Extensions

Future improvements you might add:

- Evaluation scripts (BLEU, ROUGE, METEOR) comparing `answer_text` vs. `Model Prediction`.
- Better decoding (beam search, nucleus sampling) instead of pure greedy.
- Multi-frame or full-video summarization.
- Cleaning and normalizing the dataset so that questions and answers follow a more standardized template.

