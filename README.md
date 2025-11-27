# MCD-TSF
The implementation for our paper titled Multimodal Conditioned Diffusive Time Series Forecasting.
### Requirements
To install all dependencies:
```
pip install -r requirements.txt
```
### Datasets
Public benchmark [Time-MMD](https://github.com/adityalab/time-mmd)
### Experiment
```bash
bash ./run.sh 
```

### Retrieval-augmented / CoT guidance
- `--use_rag_cot`：开启文本引导，默认使用 TF-IDF 检索 + 思维链生成中间趋势文本。
- 仅用 CoT：加 `--cot_only`（或配置中 `cot_only: true`）禁用检索，只基于数值摘要 + 原始文本生成思维链。
- 可调参数：`--rag_topk`、`--cot_model`（本地因果 LM 路径/ID，不设则用模板）、`--cot_max_new_tokens`、`--cot_temperature`、`--cot_cache_size`、`--cot_device`。
- 生成的 CoT 文本会与原始报告拼接后送入现有文本编码器，为扩散模型提供带推理的上下文。
### Acknowledgements
Codes are based on:
[CSDI](https://github.com/ermongroup/CSDI)
[Time-LLM](https://github.com/KimMeen/Time-LLM/tree/main)
[MM-TSF](https://github.com/adityalab/time-mmd)
[Autoformer](https://github.com/thuml/Autoformer)
