# Pipeline 2 → Structured Task Dataset

This repository implements a structured pipeline that converts **timestamped video tasks** into a hierarchical **mission → sub‑mission → task → subtask** representation.  
The goal is to enable **robot reasoning and LLM training** for long‑horizon tasks by ensuring logical consistency, coherent grouping, and validated execution plans.

The pipeline processes timestamped instructions extracted from videos and produces:

- logical reasoning structure
- coherent thematic grouping
- robot‑executable task blueprints
- validation logs for training data quality
- multiple visualization plots mirroring the JSON outputs

---

# System Hierarchy

The final representation follows this structure:

Mission (video)
↓
Sub‑missions (coherent themes)
↓
Tasks
↓
Subtasks (timestamped actions)

This hierarchy allows robots and AI systems to understand complex workflows at different levels of abstraction.

---

# Full Processing Pipeline

tasks_with_timestamps  
→ thread segmentation  
→ LLM logical validation  
→ category annotation  
→ graph‑based coherent regrouping  
→ sub‑mission generation  
→ validation & training quality logging  
→ task blueprint generation  
→ visualization

---

# Repository Structure

scripts/
processing and visualization scripts

prompts/
LLM prompts used for reasoning validation

results/
generated outputs from each pipeline stage

plots/
visualization outputs

---

# Step‑by‑Step Pipeline Execution

## 1. Thread Segmentation

Groups subtasks into logical reasoning threads.

Script:
scripts/build_subtask_threads.py

Run:

python scripts/build_subtask_threads.py   --input_dir results/tasks_with_timestamps   --output_dir results/subtask_threads

Output:

results/subtask_threads/<video_id>.json

Purpose:

- Identify reasoning threads across subtasks
- Prepare structured input for logical validation

---

## 2. Logical Validation (Thread Logic)

An LLM analyzes logical relationships between subtasks.

Script:

scripts/thread_logic_check.py

Prompt:

prompts/prompt_for_thread_logic.txt

Run:

python scripts/thread_logic_check.py   --model Qwen/Qwen2.5-7B-Instruct   --gpus 1   --tokens 12000   --prompt_file prompts/prompt_for_thread_logic.txt   --input_dir results/subtask_threads   --output_dir results/thread_logic

Output:

results/thread_logic/<video_id>.json

This step detects:

- logical links
- reasoning continuity
- conflicts
- repair suggestions

Visualization:

Logical Map

---

## 3. Category Annotation

Annotates subtasks with semantic categories.

Script:

scripts/categorize_tasks_and_subtasks.py

Run:

python scripts/categorize_tasks_and_subtasks.py   --input_dir results/thread_logic   --output_dir results/categorized_threads

Output:

results/categorized_threads/<video_id>.json

Categories include:

perception  
narration  
planning  
motion  

These categories help identify different reasoning roles.

---

## 4. Graph‑Based Coherent Regrouping

Subtasks are regrouped into coherent blocks using graph‑based clustering and optional LLM boundary judgment.

Scripts:

scripts/regroup_subtasks_coherence.py  
prompts/prompt_for_boundary_judge.txt

Run:

python scripts/regroup_subtasks_coherence.py   --input_dir results/categorized_threads   --output_dir results/coherent_blocks   --prompt_file prompts/prompt_for_boundary_judge.txt

Output:

results/coherent_blocks/<video_id>.json

Visualization:

scripts/plot_coherent_blocks_by_category_tree.py

Run:

python scripts/plot_coherent_blocks_by_category_tree.py   --input_dir results/coherent_blocks   --out_dir results/plots/coherent   --dpi 300

Produces:

Coherent Map

---

## 5. Sub‑Mission Generation

Each coherent block becomes a **sub‑mission** with a theme name.

Script:

scripts/add_submissions_to_coherent_blocks.py

Run:

python scripts/add_submissions_to_coherent_blocks.py   --input_dir results/coherent_blocks   --output_dir results/coherent_blocks_with_submissions   --topk 3

Output:

results/coherent_blocks_with_submissions/<video_id>.json

Example:

Mission: Soil Fertility Testing Market

Sub‑mission 1: organic / advanced / agriculture  
Sub‑mission 2: farming / fertility / food

---

## 6. Task Blueprint Generation

Creates a timestamp‑ordered execution list for robots.

Scripts:

scripts/generate_task_blueprints.py  
scripts/plot_task_blueprints_order.py

Run:

python scripts/generate_task_blueprints.py   --input_dir results/thread_logic   --output_dir results/task_blueprints

Output:

results/task_blueprints/<video_id>.json

Visualization:

python scripts/plot_task_blueprints_order.py   --input_dir results/task_blueprints   --out_dir results/plots/blueprints   --dpi 300

Produces:

Task Blueprint Plot

---

## 7. Validation Reports

Checks logical consistency and coherence.

Script:

scripts/generate_check_reports.py

Run:

python scripts/generate_check_reports.py   --thread_logic_dir results/thread_logic   --coherent_blocks_dir results/coherent_blocks_with_submissions   --output_dir results/check_reports

Outputs:

logical_check.json  
coherence_check.json

---

## 8. Training Quality Log

Records which missions and sub‑missions are suitable for training data.

Scripts:

scripts/generate_training_quality_log.py  
scripts/generate_training_quality_log_submissions.py

Run:

python scripts/generate_training_quality_log_submissions.py   --check_reports_dir results/check_reports   --thread_logic_dir results/thread_logic   --coherent_blocks_dir results/coherent_blocks_with_submissions   --task_blueprints_dir results/task_blueprints   --out_dir results/training_quality_log_submissions

Output:

training_quality_log.json  
missions/<video_id>.json

Human review decisions:

accept  
redo  
give_up  

---

## 9. Human‑in‑the‑Loop Visualization

Shows validation and correction workflow.

Script:

scripts/plot_human_in_loop.py

Run:

python scripts/plot_human_in_loop.py   --log_dir results/training_quality_log_submissions/missions   --out_dir results/plots/human_in_loop   --dpi 300

---

## 10. Mission → Sub‑Mission → Task Visualization

Displays the final hierarchy.

Script:

scripts/plot_mission_submissions_map.py

Run:

python scripts/plot_mission_submissions_map.py   --input_dir results/training_quality_log_submissions/missions   --out_dir results/plots/mission_submissions   --dpi 300

---

# Generated Visualizations

The pipeline generates several plots:

Logical Map  
Coherent Map  
Task Blueprint  
Human‑in‑the‑Loop Review  
Mission → Sub‑mission → Task Hierarchy

These visualizations mirror the JSON outputs and assist in validating reasoning structure.

---

# Expected Output Structure

results/

thread_logic/  
coherent_blocks/  
coherent_blocks_with_submissions/  
task_blueprints/  
check_reports/  
training_quality_log_submissions/  
plots/

---

# Purpose

The generated JSON structures form **high‑quality training data** for LLM‑based robotic reasoning systems.  
By validating logical consistency and coherence before training, the pipeline ensures that robots can learn **human‑level task understanding for long‑horizon missions**.
