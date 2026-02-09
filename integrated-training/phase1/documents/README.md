# Documents

This directory contains supporting documentation and report-style artifacts for the
video-to-robot-guidance pipeline.

Unlike `results/` (which stores machine-readable intermediate outputs), `documents/`
stores human-consumable materials such as HTML reports, visualizations, and notes
used for inspection, presentation, and debugging.

---

# What You’ll Find Here

## 1) HTML Reports / Visualizations

This folder includes `.html` files used to view pipeline outputs in an easy-to-read
format, such as:

- Task and subtask breakdowns
- Timestamp alignment summaries
- Frame captions
- Robot-centric guidance outputs

These HTML files provide a convenient way to inspect structured results without
opening raw JSON files.

---

## How to Open

Open the HTML files directly in a web browser:

- Double-click the file, or
- Right click → Open With → Chrome / Firefox / Edge

---

## Image Dependencies

Some HTML reports may reference images generated in:

- `results/frame_extractions/`
- `results/frame_captions/`

To ensure images load correctly:

- Keep the repository folder structure unchanged
- Open the HTML files from the local filesystem

---

## 2) Research Notes and Supporting Materials

This directory may also include:

- Experiment notes
- Design documentation
- Figures or screenshots used in presentations or papers

---

# Relationship to Other Folders

- `scripts/` generates all processing outputs
- `results/` stores structured machine-readable artifacts
- `documents/` stores presentation and inspection artifacts

The structured JSON files in `results/` remain the source of truth.

---

# Usage

These documents are intended for:

- Manual inspection of pipeline outputs
- Demonstrations
- Sharing results with collaborators or advisors
- Preparing figures and summaries for reports or papers

---

# Notes

- HTML files are typically derived from pipeline outputs
- They do not modify the underlying data
- For reproducibility, refer to the original JSON files in `results/`
