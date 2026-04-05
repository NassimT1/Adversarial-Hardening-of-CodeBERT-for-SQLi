# Adversarial-Hardening-of-CodeBERT-for-SQLi

LLM generation (+ mutation): https://github.com/tusharbhatia02/CSI5388_LLM_SQL_Injector_Generator \
AST parsing (sqlglot): https://github.com/Shortarms703/CSI5833_part1_project \
Sandboxing: https://github.com/NassimT1/CSI5388_Sandboxing \
LLM-as-a-Judge: https://github.com/saatvikpaul19/LLM-as-a-Judge 

To run the complete adversarial pipeline (Generation → AST Profiling → Sandboxing → CodeBERT Detection → LLM Judge), run the following command in CMD:
```
python run_pipeline.py
```

`codebert_training_and_results.ipynb` contains the code for retraining CodeBERT and evaluating its performance on the original kaggle dataset and our generated adversarial samples.

Download the fine-tined and hardened CodeBERT models from Google Drive: https://drive.google.com/drive/folders/1IYeOj4JRLj6e-7LIkoyHHjLmvdkWFxaF?usp=sharing

The folder `fine_tuned_codebert_model` contains the CodeBERT trained from https://github.com/boulbaba1981/SQLi-Detector. 

The folder `hardened_codebert_model` contains the CodeBERT model after retraining on our generated adversarial samples.