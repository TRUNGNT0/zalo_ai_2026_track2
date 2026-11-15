# DataMate — Zalo AI Challenge 2026, Track 2

DataMate answers Vietnamese questions about each shop's sales data (CSV, XLSX and SQLite) and returns a typed answer or
`unanswerable`. The pipeline loads and normalises each shop's data into DuckDB, prepares canonical sales/order and debt views,
then asks Qwen3.5-9B with a LoRA adapter to generate SQL. DuckDB runs the SQL samples read-only; the system repairs failed
queries, votes on the resulting answers and abstains when there is not enough agreement.

## Run inference

Requires Python 3.12+, CUDA 12.8+ and a bf16 GPU with at least 24 GB of memory. From the repository root:

1. Install Git LFS and fetch the adapter:

   ```bash
   git lfs install
   git lfs pull
   ```

2. Create an environment and install the inference dependencies:

   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -e ".[llm]"
   ```

3. Download the base model:

   ```bash
   .venv/bin/hf download Qwen/Qwen3.5-9B \
     --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
     --local-dir models/Qwen3.5-9B
   ```

4. Put the organizer's test files under `data/public_test/`:

   ```text
   data/public_test/
   ├── questions.jsonl
   ├── rulebook.md
   ├── calendar.csv
   └── data/<shop>/    # shop data files
   ```

5. Run the public test:

   ```bash
   bash scripts/run_public.sh runs/public
   ```

   By default, this reuses the saved views in `configs/public_v9`. To rebuild views from scratch:

   ```bash
   FRESH=1 bash scripts/run_public.sh runs/public_fresh
   ```

The submission is written to `runs/public/submission.jsonl`. The script merges GPU shards and checks the submission format
when the run finishes. Set `NUM_GPUS` to use a specific number of visible GPUs; model and adapter paths can be overridden
with `MODEL_DIR` and `LORA_DIR`.
