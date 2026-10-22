# Gemini Code Assist PR Review Styleguide
**Repository:** `carcinoma-binary-classification-methods` (RationAI)
**Context:** A research-focused machine learning repository for binary carcinoma classification in digital pathology WSIs across several organs (prostate, breast, colorectum). It compares setups trained under the same protocol: tile-level classifiers (CNN/ViT on tiles, or heads on frozen foundation-model embeddings such as Prov-GigaPath and Virchow2) with heuristic slide-level aggregation, attention MIL on bags of embeddings (classic = slide labels only, hybrid = slide + tile labels), and stain vs. plain augmentations. The long-term goal is to extract a reusable template of the best-performing runs, so setups must stay comparable across organs.

## 🎯 Primary Review Focus
- **Ignore formatting and linting:** We use `ruff` for formatting/linting and `uv` for package management. Assume CI/CD will catch styling issues. Do not comment on line length, quotes, or basic PEP-8 formatting.
- **Focus on ML logic and correctness:** Look for off-by-one errors in array slicing, incorrect tensor device allocations, tensor shape mismatches, incorrect masking of padded tiles in MIL bags, and data leakage between train/val/test splits (splits are by slide/patient, never by tile).
- **Comparability across setups:** Flag changes that silently alter the training/evaluation protocol for one organ, model or setup but not the others (e.g., different metrics, thresholds or sampling), unless the PR explains why.
- **Research context over production validation:** This is research code. Do not suggest adding heavy input validation, complex exception handling, or enterprise-grade defensive programming unless the current logic will explicitly crash the pipeline. Prioritize readability.
- **Testing:** Do not block PRs or aggressively request unit tests. We do not enforce strict unit testing for this repository.

## 📝 General Comment Style
- Keep comments **short and actionable**.
- Prefer **bullet points** over long paragraphs.
- Point to specific lines or sections when possible.
- Suggest improvements, not rewrite entire snippets.
- Avoid repetition of what the code already clearly states.
- Defer to the repo’s existing conventions unless there’s a clear bug or inconsistency.

## 🔬 Domain-Specific Guidance (RationAI & ratiopath)
- **Use `ratiopath`:** This project relies on our library `ratiopath`.
  - If you see custom tiling logic, suggest using `ratiopath.tiling`.
  - If you see custom annotation parsing (ASAP/GeoJSON), suggest using `ratiopath.parsers`.
  - Check if Ray-based distributed processing in `ratiopath` is being used efficiently for large-scale WSI tasks.
- **WSI Handling:** Verify that `openslide` or `ratiopath` calls use the correct downsample levels and that tile offsets are calculated correctly.
- **Weak labels:** Breast and colorectum have no carcinoma masks; the epithelium mask is used as a weak carcinoma ROI (see the organ adapters in `configs/experiment/train/`). Keep this in mind when reviewing tile-label logic.

## 🏗️ Architecture & Reproducibility
- **Hydra Configs (`configs/`):** Reproducibility is paramount. If a PR introduces a new module, model architecture, dataset or preprocessing step, check that the corresponding YAML configuration was added or updated. Remind the author if it seems missing.
  - Encoder-specific configs come in pairs (`pgp/` and `virchow2/`); flag a config added for one encoder but not the other.
  - Dataset configs (`configs/data/<organ>/...`) reference MLflow artifact URIs and optional local paths; `"..."` marks a value not produced yet.
  - Trained models are referenced from `configs/model/checkpoints/<organ>/` together with their tile and slide-level thresholds.
- **Experiment Tracking (MLflow):** When PRs add new loss functions, evaluation metrics, callbacks, or training loops in `ml/`, ensure that these new metrics are properly logged to MLflow (via `rationai.mlkit`).
- **Repository Structure:**
  - `exploration/`: Create metadata CSV for given dataset.
  - `correction/`: Contains correction scripts in case of some errors.
  - `preprocessing/`: Ensure data transformations (masks, QC, tiling, tile filtering, splits, embeddings) are logically sound. `tiling_v2/` is the current tiling; embeddings have two paths (`tile_embeddings` → `merge_embeddings`, and `tile_embeddings_v2` → `merge_embeddings_v2` → `split_embeddings`).
  - `ml/`: Focus on training loops, PyTorch Lightning modules, datamodules/samplers, model definitions, slide-level aggregators and evaluation callbacks.
  - `postprocessing/`: Focus on slide-level evaluation, aggregation and final prediction logic.
  - `notebooks/`: Focus on a one-time specific experiments.

## 📚 Types & Documentation
- **Type Hinting:** We use strict `mypy`, but it is *not required* for PRs to be merged. Gently suggest type hints for complex function signatures, but do not nitpick missing `Any` types or incomplete typing.
- **Docstrings:** Docstrings are not strictly required everywhere. However, if a docstring *is* provided, ensure it generally follows the **Google Docstring Style**.

## 💻 Libraries & Best Practices
- **PyTorch & PyTorch Lightning:** Suggest idiomatic PyTorch Lightning constructs (e.g., using `self.log` correctly). Watch out for detached tensors or memory leaks in custom training steps.
- **Data Processing (NumPy, Pandas, OpenSlide):** Suggest vectorized operations over `for` loops where applicable for performance. Ensure OpenSlide WSI coordinate extractions are logical (e.g., matching the correct level/downsample).
