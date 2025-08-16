# Vehicle Repair Summarizer & Tagging

This repository contains a Python CLI application that summarizes and categorizes auto repair records. It uses local ML models for tag predictionand can a locally hosted LLM to generate auto issue and repair description summaries.

## Pre-requisites:
- This script requires a locally hosted LLM server exposing an OpenAI-compatible chat completions endpoint at `http://<host>:<port>/engines/llama.cpp/v1/chat/completions`.

See the following ollama installation guides for setup details: 
- https://translucentcomputing.github.io/kubert-assistant-lite/ollama.html
- https://github.com/ollama/ollama/blob/main/docs/faq.md#how-do-i-configure-ollama-server
- https://github.com/ollama/ollama/blob/main/docs/quickstart.md


## Server Usage

You can run `main.py` non-interactively on a server (e.g., as a cron job or systemd service). The script reads an Excel file, produces summaries and tag predictions, and writes results to a new Excel file while writing logs to `logs/`.

Example:

```bash
python main.py \
  --input /data/claims_input.xlsx \
  --output /data/claims_output.xlsx \
  --llm-host 127.0.0.1 \
  --llm-port 12434 \
  --llm-model gemma3 \
  --log-level INFO
```

### CLI Arguments

- `--input <path>`: Path to input Excel file (required). Default in repo examples: `data/sample_input.xlsx`.
- `--output <path>`: Path to write output Excel (required). Default in repo examples: `data/sample_output.xlsx`.
- `--llm-host <host>`: Hostname/IP for the LLM API (default: `localhost`).
- `--llm-port <port>`: Port for the LLM API (default: `12434`).
- `--llm-model <name>`: LLM model name to request (default: `gemma3`).
- `--log-level <LEVEL>`: One of `DEBUG, INFO, WARNING, ERROR, CRITICAL` (default: `INFO`).



### Input File Assumptions

`main.py` expects an Excel sheet that matches the schema of `mbusa_all_claims_canonical.xlsx`.

### Output

- An Excel file at `--output` containing original and newly added summary/tag columns.
- Log file in `logs/` with a timestamped filename (e.g., `vehicle_repair_YYYYMMDD_HHMMSS.log`).

### Logging Behavior

- Logs are written to both the console and `logs/` by default.
- Log level is configurable via `--log-level`.
- Each run creates a new timestamped log file. Example entry:

```
2025-08-15 16:40:17,639 - INFO - Starting processing with input: data/sample_input.xlsx
2025-08-15 16:40:17,640 - INFO - Output will be saved to: data/sample_output.xlsx
2025-08-15 16:40:17,640 - INFO - ==================================================
2025-08-15 16:40:17,640 - INFO - Starting vehicle repair summarization
2025-08-15 16:40:17,640 - INFO - Command line arguments: /Users/johndoe/Documents/github/vehicle-repair-summarization/main.py
2025-08-15 16:40:17,915 - INFO - Spacy model loading completed in 0.28 seconds
2025-08-15 16:40:18,054 - INFO - Loaded 4 rows from data/sample_input.xlsx
2025-08-15 16:40:18,054 - INFO - Starting record summarization using local LLM
2025-08-15 16:40:18,054 - INFO - LLM Server: localhost:12434, Model: gemma3
2025-08-15 16:40:18,055 - INFO - Processing record 1/4 (ID: 1866)
2025-08-15 16:40:22,932 - INFO - Processing record 2/4 (ID: 1867)
2025-08-15 16:40:26,095 - INFO - Processing record 3/4 (ID: 1868)
2025-08-15 16:40:28,713 - INFO - Processing record 4/4 (ID: 3492)
2025-08-15 16:40:31,969 - INFO - ==================================================
2025-08-15 16:40:31,970 - INFO - Summarization completed. Processed 4 records in 0.2 minutes
2025-08-15 16:40:31,970 - INFO - Successfully processed: 4/4 records (100.0%)
2025-08-15 16:40:31,970 - INFO - Average processing time: 3.48 seconds/record
2025-08-15 16:40:31,970 - INFO - LLM summarization completed in 13.92 seconds
2025-08-15 16:40:31,970 - INFO - ==================================================
2025-08-15 16:40:31,970 - INFO - Starting prediction of enumerable fields...
2025-08-15 16:40:32,612 - INFO - ai04g__issue_presentation prediction completed in 0.64 seconds
2025-08-15 16:40:32,836 - INFO - ai04h__issue_type prediction completed in 0.22 seconds
2025-08-15 16:40:32,984 - INFO - ai04m__repair_costs_handling prediction completed in 0.15 seconds
2025-08-15 16:40:33,085 - INFO - ai04s__does_repair_fall_under_warranty prediction completed in 0.10 seconds
2025-08-15 16:40:33,237 - INFO - ai04i__issue_verified prediction completed in 0.15 seconds
2025-08-15 16:40:33,373 - INFO - ai04r__oem_engineering_services_involved prediction completed in 0.14 seconds
2025-08-15 16:40:33,526 - INFO - ai04j__repair_performed prediction completed in 0.15 seconds
2025-08-15 16:40:33,579 - INFO - ai04k___of_repairs_performed_for_this_issue prediction completed in 0.05 seconds
2025-08-15 16:40:33,626 - INFO - ai04n__not_repaired_reason prediction completed in 0.05 seconds
2025-08-15 16:40:33,732 - INFO - ai04l__is_this_issue_the_primary_issue_driving_the_days_down prediction completed in 0.11 seconds
2025-08-15 16:40:33,850 - INFO - ai04o__days_out_reason prediction completed in 0.12 seconds
2025-08-15 16:40:33,856 - INFO - ai04q__outside_influences prediction completed in 0.01 seconds
2025-08-15 16:40:33,856 - INFO - All predictions completed completed in 1.89 seconds
2025-08-15 16:40:33,856 - INFO - ==================================================
2025-08-15 16:40:33,872 - INFO - Results successfully saved to data/sample_output.xlsx
2025-08-15 16:40:33,872 - INFO - Saving results completed in 0.02 seconds
2025-08-15 16:40:33,872 - INFO - ==================================================
2025-08-15 16:40:33,872 - INFO - Total execution time: 0.27 minutes
2025-08-15 16:40:33,872 - INFO - Processing completed successfully
```


## Context

- Summarization of auto issue and repair descriptions were only tested using locally hosted LLM (e.g. gemma3) completions
- All Logistic Regression tagging models stored in models/ directory were trained from "mbusa_all_claims_canonical.xlsx"
- Provides a utility to retrain the tagging models (LogisticRegression Classifier) using train_logit_classifiers.py



## Disclaimer on Tag Prediction Accuracy

Each of the LogisticRegression Classifier models were trained on the full MBUSA_all_claims.xlsx dataset. The models are stored in the models/ directory and are utilized by main.py to predict tag values. 

Take note that while some models perform well, others do not. This is due to limitations in the training data where heavy class imbalances made it difficult to train highly accurate models. The predictions for columns marked with ❌, should be regarded with caution.


| Model | Accuracy | Macro F1 | Target Classes | Class Imbalance Risk | Production Readiness |
|-------|----------|----------|----------------|----------------------|----------------------|
| ai04g__issue_presentation | 0.862 | 0.75 | 4 | Low | ✅ Yes |
| ai04s__does_repair_fall_under_warranty | 0.878 | 0.54 | 3 | Medium | ✅ Yes |
| ai04i__issue_verified | 0.824 | 0.60 | 3 | Medium | ✅ Yes |
| ai04j__repair_performed | 0.820 | 0.60 | 3 | Medium | ✅ Yes |
| ai04n__not_repaired_reason | 0.760 | 0.52 | 6 | Medium | ✅ Yes (monitor) |
| ai04h__issue_type | 0.543 | 0.49 | 42 | High | ❌ No |
| ai04m__repair_costs_handling | 0.677 | 0.44 | 6 | High | ❌ No |
| ai04r__oem_engineering_services_involved | 0.969 | 0.69 | 2 | **Very High** | ❌ No (imbalance) |
| ai04k___of_repairs_performed_for_this_issue | 0.811 | 0.35 | 5 | High | ❌ No |
| ai04l__is_this_issue_the_primary_issue_driving_the_days_down | 0.569 | 0.53 | 3 | Medium | ❌ No |
| ai04o__days_out_reason | 0.483 | 0.33 | 7 | High | ❌ No |
| ai04q__outside_influences | 0.800 | N/A | 2 | **Very High** (tiny test set) | ❌ No (tiny test set) |
