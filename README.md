# T5-Base Fine-Tuning for English to Hinglish Generation

This repository fine-tunes **T5-base** for English-to-Hinglish code-mixed text generation using the HinGE dataset (HinglishEval INLG 2022 Shared Task).

This implementation accompanies:

*Code-Mixer Ya Nahi: Novel Approaches To Measuring Multilingual LLMs' Code-Mixing Capabilities*  
Ayushman Gupta*, Akhil Bhogal*, Kripabandhu Ghosh  
IISER Kolkata  
https://arxiv.org/abs/2410.11079

---

## Task

Given an English sentence, the model generates a Hinglish (Hindi–English code-mixed) sentence.

During training, inputs are prefixed with: "Generate Hinglish from English:"

## Expected Directory Structure
```
Data/HinglishEval_INLG_2022_shared_task/
      train.csv
      valid.csv
      test.csv
Split_hinge_data/
```


All splits are combined and re-split internally:

- Total samples: 13,738  
- Train: 11,677  
- Validation: 2,061  
- Validation split: 15% (shuffled)

---

## Model and Training Configuration

### Base Model

- `t5-base`

### Hyperparameters

- Learning rate: 2e-5  
- Batch size (train): 8  
- Batch size (eval): 8  
- Epochs: 20  
- Weight decay: 0.001  
- FP16: True  
- Evaluation strategy: per epoch  
- Predict with generate: True  

### Sequence Lengths

- Max input length: 210  
- Max target length: 450  

### Hardware

- NVIDIA Tesla P100 (16GB VRAM)

---

## Validation Performance (HinGE)

Best validation metrics:

- ROUGE-1: 32.78  
- ROUGE-L: 31.33  
- Validation loss: ≈ 2.02  

A ROUGE curve is saved as: `rouge_curve.png`

---

## Results on EMNLP 2023 Zhang et al Dataset

Evaluated on the dataset from:  
https://aclanthology.org/2023.emnlp-main.774/

- BLEU: 13.97  
- ROUGE-L (F1): 36.21  
- METEOR: 35.12  

---

## Citation

If you use this code, please cite
```bibtex
@misc{gupta2024codemixeryanahinovel,
      title={Code-Mixer Ya Nahi: Novel Approaches to Measuring Multilingual LLMs' Code-Mixing Capabilities}, 
      author={Ayushman Gupta and Akhil Bhogal and Kripabandhu Ghosh},
      year={2024},
      eprint={2410.11079},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.11079}, 
}
```
