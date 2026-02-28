## Official code accompanying the paper:

> **Code-Mixer Ya Nahi: Novel Approaches to Measuring Multilingual LLMs' Code-Mixing Capabilities**
> Ayushman Gupta\*, Akhil Bhogal\*, Kripabandhu Ghosh
> IISER Kolkata
> [[arXiv]](https://arxiv.org/abs/2410.11079)

---

## Overview 

This repository implements fine-tuning of T5-base on the PHINC dataset. 

## Hyperparameters and Details

### Training

Total Sentences: 13738 -> Split 0.15 fraction (shuffled) with Valid Data.

Train=11677, Valid=2061

Prefix=“Generate Hinglish from English:” was added to input. 

Max Input Length = 114

Max Target Length = 118

Optimizer: AdamW

fp16 = False

Per Device Train batch size = 24

Per Device Eval Batch Size = 6

Learning Rate = 5e-5

Train Epochs = 30

GPU: NVIDIA Tesla P100 (16GB VRAM)


### Inference
Temperature = 0.001

Repetition Penalty = 2.0

---

## Citation

If you use this code, please cite: 

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
