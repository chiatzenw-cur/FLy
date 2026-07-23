### 1. Running FLy:
Replaced llama 70B with Qwen 1.5 and 14B beacuse of GPU memory constraints

### 2. Baselines:

### Standard SD without FLy:
```
Running generate_until requests:  99%|█████████▉| 163/164 [29:27<00:11, 11.04s/it]2026-07-23:01:01:34 INFO     [models.huggingface:1215] Speed:14.97|MAT:9.48|ngramMAT:0.00|DraftRound:3|TotalTok:12|Elapsed:2.87|InitialMismatch:15|FLyAccepted:0|FinalRejected:15|FLyAcceptRate:0.00%
Running generate_until requests: 100%|██████████| 164/164 [29:31<00:00, 10.80s/it]
2026-07-23:01:01:34 INFO     [models.huggingface:1277] ================================================================================
2026-07-23:01:01:34 INFO     [models.huggingface:1278] Global Statistics
2026-07-23:01:01:34 INFO     [models.huggingface:1279] ================================================================================
2026-07-23:01:01:34 INFO     [models.huggingface:1280] Total Samples: 164
2026-07-23:01:01:34 INFO     [models.huggingface:1281] 
2026-07-23:01:01:34 INFO     [models.huggingface:1282] Totals:
2026-07-23:01:01:34 INFO     [models.huggingface:1283]   Initial Mismatch: 5371
2026-07-23:01:01:34 INFO     [models.huggingface:1284]   FLy Accepted: 0
2026-07-23:01:01:34 INFO     [models.huggingface:1285]   Final Rejected: 5371
2026-07-23:01:01:34 INFO     [models.huggingface:1286]   FLy Accept Rate: 0.00%
2026-07-23:01:01:34 INFO     [models.huggingface:1287] 
2026-07-23:01:01:34 INFO     [models.huggingface:1288] Averages per Sample:
2026-07-23:01:01:34 INFO     [models.huggingface:1289]   Avg Initial Mismatch: 32.75
2026-07-23:01:01:34 INFO     [models.huggingface:1290]   Avg FLy Accepted: 0.00
2026-07-23:01:01:34 INFO     [models.huggingface:1291]   Avg Final Rejected: 32.75
2026-07-23:01:01:34 INFO     [models.huggingface:1292]   Avg FLy Accept Rate: 0.00%
2026-07-23:01:01:34 INFO     [models.huggingface:1293] ================================================================================
2026-07-23:01:02:44 INFO     [loggers.evaluation_tracker:209] Saving results aggregated
2026-07-23:01:02:44 INFO     [loggers.evaluation_tracker:298] Saving per-sample results for: humaneval_instruct
hf (pretrained=Qwen/Qwen2.5-Coder-1.5B-Instruct,config_path=fly_config/FLy_Qwen25Coder_14b.json,enable_statistics=true,total_gen_tok=256,enable_fly=false,use_ngram=false,entropy_thre=0), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 1
|      Tasks       |Version|  Filter   |n-shot|Metric|   |Value |   |Stderr|
|------------------|------:|-----------|-----:|------|---|-----:|---|-----:|
|humaneval_instruct|      4|create_test|     0|pass@1|   |0.8963|±  |0.0239|
```