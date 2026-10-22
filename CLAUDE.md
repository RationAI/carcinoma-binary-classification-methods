# CLAUDE.md

## Purpose

Research repo (RationAI, MUNI) for **binary carcinoma classification of WSIs** (whole slide images) across organs: **prostate, breast, colorectum**. Several setups are trained under the same protocol and compared:

- **Tile-level (TL) classifiers + heuristic slide aggregation**: CNN/ViT on image tiles, or a linear head on frozen foundation-model embeddings. Slide scores come from aggregating tile predictions (max, mean-pool-max, top-k, span).
- **MIL on bags of embeddings**: attention MIL over all tile embeddings of a slide. *Classic* = slide-level (SL) labels only; *hybrid* = SL + TL labels.
- **Stain augmentations** (`*_stain` variants) vs plain augmentations, plus 512 px vs 224 px tiles for CNNs.

Long-term goal: find what works best and extract a **reusable template of good runs**. Keep setups comparable across organs: same structure, naming and metrics. Don't add one-off divergences.

`.gemini/styleguide.md` is the PR-review guide for Gemini Code Assist; keep it in sync with this file.

## Layout

| Dir | What |
|---|---|
| `ml/` | Lightning training/test/predict. Entry point `python -m ml`. Models: `base_model.py` (TL base), `cnn_model.py`, `vit_model.py`, `embedding_model.py`, `foundation_model.py`, `mil_model_base.py` + `classic_mil_model.py` / `hybrid_mil_model.py`. Plus `datamodule/`, `modeling/backbone` (resnet50, vgg16, google_vit, pgp, virchow2), `modeling/decode_head`, `aggregators/`, and `callbacks/` (test-time metrics, curves, histograms, heatmaps, prediction tables) |
| `preprocessing/` | Hydra scripts: masks (tissue, QC, annotation), tiling (`tiling_v2/` is current, `tiling/` is legacy), tile filtering, stats, splits, embeddings (two parallel paths, `tile_embeddings` and `tile_embeddings_v2`; see Data pipeline) |
| `postprocessing/` | Slide-level evaluation from prediction tables (`slide_level_eval`, `slide_level_curves`, `slide_prediction_table[_merge]`, `add_auprc`, `eval_estimation`) |
| `exploration/` | Per-dataset scripts that build the metadata CSV |
| `correction/` | One-off fix-up scripts for broken metadata/tiles |
| `notebooks/` | One-off analyses |
| `configs/` | Hydra configs. Most of the work is here (~1200 experiment YAMLs) |

## Running

Everything runs via `uv` from the repo root. Each script has a root Hydra config and an `+experiment=` selection:

```bash
uv run python -m ml mode=fit  experiment=train/embeddings/virchow2/c20_24
uv run python -m ml mode=test experiment=test/embeddings/virchow2/test_suite/full_breast_test
uv run python -m preprocessing.embeddings.store_embeddings +experiment=preprocessing/store_embeddings/pgp/c20_24_val
uv run python -m postprocessing.slide_level_eval +experiment=postprocessing/slide_level_eval/virchow2/top_k_aggregations/mmci_sl_group_1
```

- `ml` uses `configs/ml.yaml` with `mode` ∈ {`fit`, `validate`, `test`, `predict`} (calls `trainer.<mode>`). `experiment=` has no `+`.
- Pre/postprocessing scripts use `@with_cli_args(["+preprocessing=<step>"])` with `configs/preprocessing.yaml` / `postprocessing.yaml`, and take `+experiment=...`.
- Every run logs to **MLflow** (`MLFLOW_TRACKING_URI`, via `rationai.mlkit` `@autolog`). The run name is built from `model_name` / `use_case` / `data_name`. The MLflow experiment is set per organ in `configs/data/<organ>/base.yaml` (Prostate, Breast Cancer = id 61, Colorectal cancer = id 74).

Lint/type-check (CI runs the RationAI lint workflow on PRs to `master`):

```bash
uv run ruff check . && uv run ruff format . && uv run mypy .
```

There are no unit tests. Commits use Conventional Commits (`feat: ...`, `fix: ...`); feature branches go to `master` via PR.

## Config conventions

- **Data** (`configs/data/<organ>/<source>/<dataset>.yaml`): MLflow artifact URIs (`mlflow-artifacts:/<exp>/<run>/artifacts/...`) and optional local copies (`*_path`) for every stage. Examples: `tiles_filtered_uri_512/224`, `tiles_filtered_w_{virchow2,pgp}_uri_224` (sharded embeddings) with `..._path_224`, masks, `metadata_table`, `thresholds`, `data_name`. `"..."` marks a value not produced yet. `use_paths: true` (in `ml.yaml`) makes datasets read local paths instead of URIs when both are set. Organ `base.yaml` sets `carcinoma_artifacts` (`/mnt/projects/{prostate_cancer/cancer,breast_cancer/bc,colorectal_cancer}`) and `hf_cache`.
- **Datasets**: prostate (MMCI `tl_*` = tile-annotated train/val/test, `sl*` = slide-level groups, Radboud/PANDA incl. stain-normalized test variants, Karolinska, MUG, FN Brno, Comenius); breast (MMCI `b20_24_*`, full test, mammaprint, BRACS); colorectum (MMCI `c20_24_*`, full test, ADOPT; METU, IMP-CRS-24).
- **Train experiments** (`experiment/train/{images,embeddings,bag_of_embeddings}/<model>/<dataset>.yaml`): each chains `base[_stain][_224]` + an organ adapter + `@train_data` / `@val_data`. Breast and colorectum have no carcinoma masks, so their adapters (`train/{breast,colorectum}_adapter[_mil].yaml`) use the weak epithelium mask as the carcinoma ROI and sample `num_slides` per epoch.
- **Model checkpoints** (`configs/model/checkpoints/<organ>/<model>.yaml`): trained checkpoint URI, `model_name`, the tile threshold and per-aggregator slide thresholds (`slide_{max,mpm,top_k,span}_threshold`), and the train stats used at test time. Test and postprocessing configs pull these in as `@model_data`.
- **Test experiments** (`experiment/test/<setup>/<model>/test_suite/<dataset>.yaml`) = `base_<organ>` (checkpoint) + `@test_data`. `callbacks/combined_test_suite[_mil].yaml` attaches all evaluation callbacks.
- Encoder variants come in pairs. Mirror every new `pgp/` config in `virchow2/` (and vice versa) unless told otherwise.
- Config files start with `# @package _global_`, and file names use snake_case dataset ids (`c20_24_train`, `colorectum_mmci_full_test`).

## Data pipeline (typical order for a new dataset)

`exploration/<dataset>` (metadata CSV) → `tissue_masks` / `qc_masks` / `annot_masks` → `tiling_v2` → `filter_tiles` (`tile_filters`, `overlappers`) → `mean_and_std` → split (`group_split` / `simple_split` / `negative_split`) → **embeddings** (one of the two paths below) → `store_embeddings` (download from MLflow to `/mnt/projects/...`) → `verify_embeddings` → train → test → postprocessing.

Embeddings are computed from `data.tiles_filtered_uri_224` per encoder (pgp, virchow2). Both paths can be sharded over slides with `start`/`end` (e.g. `b20_24_train_1..4`).

- **`tile_embeddings` path** (used for the MMCI prostate/breast/colorectum `*_sharded` artifacts): `tile_embeddings` runs the encoder locally on GPU (downloads it from Hugging Face, so needs `HF_TOKEN`) and writes one `<slide>.pt` tensor per slide to `${carcinoma_artifacts}/tile_embeddings/<encoder>/<data_name>` (the data config's `<encoder>_embeddings_filtered_path`). Then `merge_embeddings` attaches them to the filtered tiles parquet, writes sharded parquets and logs them as `<data_name>_sharded` (→ `tiles_filtered_w_<encoder>_uri_224`). `filter_embeddings` is a migration step that re-filters existing per-slide `.pt` embeddings after the tiles were re-filtered (prostate MMCI).
- **`tile_embeddings_v2` path**: `tile_embeddings_v2` calls the deployed RationAI model service (`rationai.AsyncClient`, `models.embed_image`, with `concurrency` parallel requests). It needs no `HF_TOKEN` or local GPU, and writes the parquet tiles + slides format directly, one `shard_*` dir per `start`/`end` run. `merge_embeddings_v2` merges the shards into one flat `tiles/` + `slides/` artifact. `split_embeddings` optionally splits a combined embeddings set into per-split datasets by slide (e.g. `b20_24_val_neg_test`).

## Gotchas

- MLflow client is 3.x but the server is older. `ml/_mlflow_compat.py` patches logged-model lookups, so keep `apply_mlflow_compat_patch()` in the entry points.
- Big artifact downloads (`store_embeddings`) use up to 20 parallel streams and can fail with "Response ended prematurely" on ~500 MB shards. Re-running re-downloads everything.
- Foundation models (pgp = Prov-GigaPath, virchow2) are pulled from Hugging Face when run locally (`tile_embeddings`, `verify_embeddings`, `train/images/{pgp,virchow2}`). `tile_embeddings_v2` uses the model service instead. Datasets cache to the organ's `hf_cache` (`ml/hf_cache.py`).
- This is research code: favour readable, config-driven changes over defensive code. When adding a module or step, add its Hydra config too.
