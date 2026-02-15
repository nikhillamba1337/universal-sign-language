# PBSS Universal Alphabet Pipeline

This repository trains/evaluates three independent classifiers for ASL, BSL, and ISL alphabets, computes per-letter macro F1 scores, and uses PBSS (Per-letter Best Source Selection) to pick the language with the highest F1 for each letter to form a universal alphabet. It also builds summary tables and a `new_sl/` dataset containing one best image per letter.

## Quick Start

1. Ensure datasets are in `datasets/{asl,bsl,isl}/{A..Z}/` and models (optional) in `models/`.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the pipeline (eval existing models only):

```bash
python pbss_pipeline.py --eval-only
```

4. To train (or retrain) and then evaluate:

```bash
python pbss_pipeline.py --train --epochs 10
```

5. Speed options (use a fraction of data or cap samples):

```bash
python pbss_pipeline.py --eval-only --sample-ratio 0.2
python pbss_pipeline.py --eval-only --max-samples 200
```

## Outputs

- `results/metrics_table.csv`: Per-letter metrics across datasets.
- `results/per_letter_f1.csv`: Per-letter F1 (%) table (ASL/BSL/ISL).
- `results/selected_letters_summary.csv`: PBSS best dataset per letter + F1.
- `results/selected_language_per_letter.csv`: Selected language for each letter.
- `new_sl/{A..Z}/`: Universal alphabet (one image per letter).
- `results/new_sl_contents.csv`: Contents of `new_sl/` for quick inspection.

## Existing Interactive Script

You can also use `main.py` if you prefer interactive mode with prompts for training/evaluation and checking the `new_sl/` dataset.
