# Adversarial-Hardening-of-CodeBERT-for-SQLi

LLM generation (+ mutation): https://github.com/tusharbhatia02/CSI5388_LLM_SQL_Injector_Generator \
AST parsing (sqlglot): https://github.com/Shortarms703/CSI5833_part1_project \
Sandboxing: https://github.com/NassimT1/CSI5388_Sandboxing \
LLM-as-a-Judge: https://github.com/saatvikpaul19/LLM-as-a-Judge 

To run the complete adversarial pipeline (Generation → AST Profiling → Sandboxing → CodeBERT Detection → LLM Judge), run the following command in CMD:
```
python run_pipeline.py
```

(ensure to include model.safetensors for CodeBERT from https://github.com/boulbaba1981/SQLi-Detector)