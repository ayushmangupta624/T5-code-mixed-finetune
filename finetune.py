# -*- coding: utf-8 -*-
"""
Fine-tuning T5-base for English-to-Hinglish Generation
=======================================================
Code accompanying the paper:

  Code-Mixer Ya Nahi: Novel Approaches To Measuring Multilingual
  LLMs' Code-Mixing Capabilities
  Ayushman Gupta*, Akhil Bhogal*, Kripabandhu Ghosh
  IISER Kolkata
  https://arxiv.org/abs/2410.11079

This script fine-tunes T5-base on the HinGE dataset to generate
Hinglish (Hindi-English code-mixed) sentences from English inputs.

Dataset paths:
    Data/HinglishEval_INLG_2022_shared_task/train.csv
    Data/HinglishEval_INLG_2022_shared_task/valid.csv
    Data/HinglishEval_INLG_2022_shared_task/test.csv
    Split_hinge_data/   (directory must exist for saving train/valid splits)
"""

import json
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from evaluate import load
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import nltk
import transformers

print(transformers.__version__)

"""LOADING THE DATASET"""

# Load HinGE train/valid/test splits
hinge_train = pd.read_csv('Data/HinglishEval_INLG_2022_shared_task/train.csv')
hinge_valid = pd.read_csv('Data/HinglishEval_INLG_2022_shared_task/valid.csv')
hinge_test  = pd.read_csv('Data/HinglishEval_INLG_2022_shared_task/test.csv')

hinge_train_eng  = json.loads((hinge_train.iloc[:,0]).to_json(orient='records'))
hinge_train_hing = json.loads((hinge_train.iloc[:,2]).to_json(orient='records'))

hinge_valid_eng  = json.loads((hinge_valid.iloc[:,0]).to_json(orient='records'))
hinge_valid_hing = json.loads((hinge_valid.iloc[:,2]).to_json(orient='records'))

hinge_test_eng  = json.loads((hinge_test.iloc[:,0]).to_json(orient='records'))
hinge_test_hing = json.loads((hinge_test.iloc[:,2]).to_json(orient='records'))

hinge_eng  = hinge_test_eng  + hinge_valid_eng  + hinge_train_eng
hinge_hing = hinge_test_hing + hinge_valid_hing + hinge_train_hing

# if using hinge only
train_eng = []
train_hing = []

for i in range(len(hinge_eng)):
    train_eng.append((hinge_eng[i]).strip())
    train_hing.append(hinge_hing[i].strip())

"""BUILDING DATASET"""

dataset_structure = {
    'Eng':  train_eng,
    'Hing': train_hing,
}

Train_dataset = Dataset.from_dict(dataset_structure)
print(Train_dataset)

train_data, val_data = train_test_split(Train_dataset, test_size=0.15, shuffle='True')

with open('Split_hinge_data/train.json', 'w', encoding='utf-8') as json_file:
    json.dump(train_data, json_file, ensure_ascii=False, indent=2)

with open('Split_hinge_data/valid.json', 'w', encoding='utf-8') as json_file:
    json.dump(val_data, json_file, ensure_ascii=False, indent=2)

with open('Split_hinge_data/train.json', 'r', encoding='utf-8') as json_file:
    train_data = json.load(json_file)
with open('Split_hinge_data/valid.json', 'r', encoding='utf-8') as json_file:
    val_data = json.load(json_file)

dataset_structure = {
    'Eng':  train_data['Eng'],
    'Hing': train_data['Hing'],
}
Train_dataset = Dataset.from_dict(dataset_structure)
print(Train_dataset)

dataset_structure2 = {
    'Eng':  val_data['Eng'],
    'Hing': val_data['Hing'],
}
Valid_dataset = Dataset.from_dict(dataset_structure2)
print(Valid_dataset)

"""PREPROCESSING"""

model_checkpoint = 't5-base-finetuned-hinge/'

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

prefix = "Generate Hinglish from English: "

if model_checkpoint in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
    prefix = "Generate Hinglish from English: "
else:
    prefix = ""

max_input_length = 210
max_target_length = 450

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["Eng"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(text_target=examples["Hing"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = Train_dataset.map(preprocess_function, batched=True)
tokenized_valid = Valid_dataset.map(preprocess_function, batched=True)

"""FINE-TUNING THE MODEL"""

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

batch_size = 8
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned-hinge2",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.001,
    save_total_limit=6,
    num_train_epochs=20,
    predict_with_generate=True,
    fp16=True,
    #push_to_hub=True,
    logging_dir="./logs",  # Directory for TensorBoard logs
    logging_steps=100
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

metric = load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds  = ["\n".join(nltk.sent_tokenize(pred.strip()))  for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    # Note that other metrics may not have a `use_aggregator` parameter
    # and thus will return a list, computing a metric for each sentence.
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=True)
    # Extract a few results
    result = {key: value * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

device_count = torch.cuda.device_count()
if device_count > 0:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
model.to(device)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model()

"""TRAINING RESULTS"""

training_stats = [
    {"loss": 3.1208, "learning_rate": 1.9536312849162013e-05, "epoch": 0.93},
    {"loss": 2.7646, "learning_rate": 1.9071694599627562e-05, "epoch": 1.86},
    {"loss": 2.6301, "learning_rate": 1.860614525139665e-05,  "epoch": 2.79},
    {"loss": 2.5484, "learning_rate": 1.8140595903165736e-05, "epoch": 3.72},
    {"loss": 2.4631, "learning_rate": 1.7675046554934827e-05, "epoch": 4.66},
    {"loss": 2.4215, "learning_rate": 1.720949720670391e-05,  "epoch": 5.59},
    {"loss": 2.3551, "learning_rate": 1.6743947858473e-05,    "epoch": 6.52},
    {"loss": 2.3144, "learning_rate": 1.6278398510242088e-05, "epoch": 7.45},
    {"loss": 2.2816, "learning_rate": 1.5812849162011175e-05, "epoch": 8.38},
    {"loss": 2.2359, "learning_rate": 1.5347299813780262e-05, "epoch": 9.31},
    {"loss": 2.2013, "learning_rate": 1.4881750465549349e-05, "epoch": 10.24},
    {"loss": 2.1646, "learning_rate": 1.4416201117318438e-05, "epoch": 11.17},
    {"loss": 2.1345, "learning_rate": 1.3950651769087525e-05, "epoch": 12.1},
    {"loss": 2.1106, "learning_rate": 1.3486033519553074e-05, "epoch": 13.04},
    {"loss": 2.0821, "learning_rate": 1.3020484171322161e-05, "epoch": 13.97},
    {"loss": 2.0638, "learning_rate": 1.255493482309125e-05,  "epoch": 14.9},
    {"loss": 2.0308, "learning_rate": 1.2089385474860335e-05, "epoch": 15.83},
    {"loss": 2.0167, "learning_rate": 1.1623836126629424e-05, "epoch": 16.76},
    {"loss": 1.9811, "learning_rate": 1.1159217877094973e-05, "epoch": 17.69},
    {"loss": 1.9651, "learning_rate": 1.0694599627560522e-05, "epoch": 18.62},
    {"loss": 1.9361, "learning_rate": 1.0229050279329609e-05, "epoch": 19.55},
    {"loss": 1.9285, "learning_rate": 9.763500931098698e-06,  "epoch": 20.48},
    {"loss": 1.8994, "learning_rate": 9.297951582867785e-06,  "epoch": 21.42},
    {"loss": 1.8971, "learning_rate": 8.832402234636872e-06,  "epoch": 22.35},
    {"loss": 1.899,  "learning_rate": 8.366852886405959e-06,  "epoch": 23.28},
    {"loss": 1.8802, "learning_rate": 7.901303538175046e-06,  "epoch": 24.21},
    {"loss": 1.8771, "learning_rate": 7.435754189944134e-06,  "epoch": 25.14},
    {"loss": 1.8567, "learning_rate": 6.970204841713222e-06,  "epoch": 26.07},
    {"loss": 1.8554, "learning_rate": 6.505586592178772e-06,  "epoch": 27.0},
    {"loss": 1.843,  "learning_rate": 6.040968342644321e-06,  "epoch": 27.93},
    {"loss": 1.8363, "learning_rate": 5.575418994413409e-06,  "epoch": 28.86},
    {"loss": 1.8223, "learning_rate": 5.109869646182496e-06,  "epoch": 29.8},
    {"loss": 1.817,  "learning_rate": 4.644320297951583e-06,  "epoch": 30.73},
    {"loss": 1.8182, "learning_rate": 4.178770949720671e-06,  "epoch": 31.66},
    {"loss": 1.817,  "learning_rate": 3.713221601489758e-06,  "epoch": 32.59},
    {"loss": 1.8034, "learning_rate": 3.248603351955307e-06,  "epoch": 33.52},
    {"loss": 1.8001, "learning_rate": 2.783054003724395e-06,  "epoch": 34.45},
    {"loss": 1.7876, "learning_rate": 2.3175046554934826e-06, "epoch": 35.38},
    {"loss": 1.7805, "learning_rate": 1.85195530726257e-06,   "epoch": 36.31},
    {"loss": 1.7814, "learning_rate": 1.3864059590316575e-06, "epoch": 37.24},
    {"loss": 1.7825, "learning_rate": 9.208566108007449e-07,  "epoch": 38.18},
    {"loss": 1.7805, "learning_rate": 4.5530726256983245e-07, "epoch": 39.11},
    {"loss": 1.7825, "learning_rate": 4.5530726256983245e-07, "epoch": 40.0},
]

validation_stats = [
    {'eval_loss': 2.630547046661377,  'eval_rouge1': 24.0904, 'eval_rouge2': 10.9732, 'eval_rougeL': 23.1353, 'eval_rougeLsum': 23.0913, 'eval_gen_len': 16.8932, 'eval_runtime': 94.6831,  'eval_samples_per_second': 17.004, 'eval_steps_per_second': 1.426, 'epoch': 1.0},
    {'eval_loss': 2.494086742401123,  'eval_rouge1': 24.781,  'eval_rouge2': 11.5239, 'eval_rougeL': 23.879,  'eval_rougeLsum': 23.8573, 'eval_gen_len': 16.8901, 'eval_runtime': 94.2496,  'eval_samples_per_second': 17.082, 'eval_steps_per_second': 1.432, 'epoch': 2.0},
    {'eval_loss': 2.4109129905700684, 'eval_rouge1': 26.0876, 'eval_rouge2': 12.3056, 'eval_rougeL': 25.1157, 'eval_rougeLsum': 25.0735, 'eval_gen_len': 16.795,  'eval_runtime': 96.0765,  'eval_samples_per_second': 16.757, 'eval_steps_per_second': 1.405, 'epoch': 3.0},
    {'eval_loss': 2.3524200916290283, 'eval_rouge1': 26.0376, 'eval_rouge2': 11.9784, 'eval_rougeL': 25.0803, 'eval_rougeLsum': 25.0432, 'eval_gen_len': 16.8839, 'eval_runtime': 98.2129,  'eval_samples_per_second': 16.393, 'eval_steps_per_second': 1.375, 'epoch': 4.0},
    {'eval_loss': 2.3063082695007324, 'eval_rouge1': 27.4421, 'eval_rouge2': 12.7481, 'eval_rougeL': 26.4719, 'eval_rougeLsum': 26.4285, 'eval_gen_len': 16.7429, 'eval_runtime': 98.9732,  'eval_samples_per_second': 16.267, 'eval_steps_per_second': 1.364, 'epoch': 5.0},
    {'eval_loss': 2.2678709030151367, 'eval_rouge1': 28.3603, 'eval_rouge2': 13.1509, 'eval_rougeL': 27.3745, 'eval_rougeLsum': 27.336,  'eval_gen_len': 16.6323, 'eval_runtime': 96.9325,  'eval_samples_per_second': 16.61,  'eval_steps_per_second': 1.393, 'epoch': 6.0},
    {'eval_loss': 2.2364509105682373, 'eval_rouge1': 28.825,  'eval_rouge2': 13.2033, 'eval_rougeL': 27.7444, 'eval_rougeLsum': 27.7146, 'eval_gen_len': 16.6689, 'eval_runtime': 99.1618,  'eval_samples_per_second': 16.236, 'eval_steps_per_second': 1.361, 'epoch': 7.0},
    {'eval_loss': 2.2083652019500732, 'eval_rouge1': 29.661,  'eval_rouge2': 13.7929, 'eval_rougeL': 28.5306, 'eval_rougeLsum': 28.455,  'eval_gen_len': 16.5702, 'eval_runtime': 99.9516,  'eval_samples_per_second': 16.108, 'eval_steps_per_second': 1.351, 'epoch': 8.0},
    {'eval_loss': 2.1830620765686035, 'eval_rouge1': 29.6742, 'eval_rouge2': 13.8306, 'eval_rougeL': 28.4792, 'eval_rougeLsum': 28.44,   'eval_gen_len': 16.6857, 'eval_runtime': 100.498,  'eval_samples_per_second': 16.02,  'eval_steps_per_second': 1.343, 'epoch': 9.0},
    {'eval_loss': 2.164599657058716,  'eval_rouge1': 29.5359, 'eval_rouge2': 13.5363, 'eval_rougeL': 28.382,  'eval_rougeLsum': 28.3629, 'eval_gen_len': 16.6491, 'eval_runtime': 100.0783, 'eval_samples_per_second': 16.087, 'eval_steps_per_second': 1.349, 'epoch': 10.0},
    {'eval_loss': 2.1482534408569336, 'eval_rouge1': 30.3119, 'eval_rouge2': 14.0193, 'eval_rougeL': 28.9716, 'eval_rougeLsum': 28.9607, 'eval_gen_len': 16.6118, 'eval_runtime': 93.9284,  'eval_samples_per_second': 17.141, 'eval_steps_per_second': 1.437, 'epoch': 11.0},
    {'eval_loss': 2.131849765777588,  'eval_rouge1': 30.3768, 'eval_rouge2': 13.9405, 'eval_rougeL': 28.993,  'eval_rougeLsum': 28.9682, 'eval_gen_len': 16.6292, 'eval_runtime': 99.1942,  'eval_samples_per_second': 16.231, 'eval_steps_per_second': 1.361, 'epoch': 12.0},
    {'eval_loss': 2.124645233154297,  'eval_rouge1': 31.0272, 'eval_rouge2': 14.4249, 'eval_rougeL': 29.6014, 'eval_rougeLsum': 29.5785, 'eval_gen_len': 16.6093, 'eval_runtime': 95.7933,  'eval_samples_per_second': 16.807, 'eval_steps_per_second': 1.409, 'epoch': 13.0},
    {'eval_loss': 2.108619451522827,  'eval_rouge1': 30.7552, 'eval_rouge2': 14.3304, 'eval_rougeL': 29.4344, 'eval_rougeLsum': 29.4312, 'eval_gen_len': 16.6025, 'eval_runtime': 99.5915,  'eval_samples_per_second': 16.166, 'eval_steps_per_second': 1.356, 'epoch': 14.0},
    {'eval_loss': 2.098215341567993,  'eval_rouge1': 30.5716, 'eval_rouge2': 14.1721, 'eval_rougeL': 29.2878, 'eval_rougeLsum': 29.2816, 'eval_gen_len': 16.6752, 'eval_runtime': 100.8761, 'eval_samples_per_second': 15.96,  'eval_steps_per_second': 1.338, 'epoch': 15.0},
    {'eval_loss': 2.0888521671295166, 'eval_rouge1': 30.8659, 'eval_rouge2': 14.259,  'eval_rougeL': 29.5423, 'eval_rougeLsum': 29.5702, 'eval_gen_len': 16.618,  'eval_runtime': 97.6623,  'eval_samples_per_second': 16.485, 'eval_steps_per_second': 1.382, 'epoch': 16.0},
    {'eval_loss': 2.080385208129883,  'eval_rouge1': 31.3382, 'eval_rouge2': 14.5159, 'eval_rougeL': 30.0226, 'eval_rougeLsum': 30.0594, 'eval_gen_len': 16.5963, 'eval_runtime': 97.4864,  'eval_samples_per_second': 16.515, 'eval_steps_per_second': 1.385, 'epoch': 17.0},
    {'eval_loss': 2.0738346576690674, 'eval_rouge1': 30.852,  'eval_rouge2': 14.4056, 'eval_rougeL': 29.583,  'eval_rougeLsum': 29.6159, 'eval_gen_len': 16.6379, 'eval_runtime': 97.219,   'eval_samples_per_second': 16.561, 'eval_steps_per_second': 1.389, 'epoch': 18.0},
    {'eval_loss': 2.0663774013519287, 'eval_rouge1': 31.3511, 'eval_rouge2': 14.5032, 'eval_rougeL': 30.0258, 'eval_rougeLsum': 30.0795, 'eval_gen_len': 16.595,  'eval_runtime': 94.6793,  'eval_samples_per_second': 17.005, 'eval_steps_per_second': 1.426, 'epoch': 19.0},
    {'eval_loss': 2.060537815093994,  'eval_rouge1': 31.7063, 'eval_rouge2': 14.7286, 'eval_rougeL': 30.3606, 'eval_rougeLsum': 30.3754, 'eval_gen_len': 16.6025, 'eval_runtime': 94.8342,  'eval_samples_per_second': 16.977, 'eval_steps_per_second': 1.424, 'epoch': 20.0},
    {'eval_loss': 2.055302143096924,  'eval_rouge1': 31.9853, 'eval_rouge2': 14.7642, 'eval_rougeL': 30.5244, 'eval_rougeLsum': 30.5514, 'eval_gen_len': 16.5832, 'eval_runtime': 96.8361,  'eval_samples_per_second': 16.626, 'eval_steps_per_second': 1.394, 'epoch': 21.0},
    {'eval_loss': 2.0542237758636475, 'eval_rouge1': 32.0725, 'eval_rouge2': 14.7372, 'eval_rougeL': 30.6596, 'eval_rougeLsum': 30.6558, 'eval_gen_len': 16.5845, 'eval_runtime': 97.3386,  'eval_samples_per_second': 16.54,  'eval_steps_per_second': 1.387, 'epoch': 22.0},
    {'eval_loss': 2.041106700897217,  'eval_rouge1': 31.7199, 'eval_rouge2': 14.6084, 'eval_rougeL': 30.4182, 'eval_rougeLsum': 30.4325, 'eval_gen_len': 16.6342, 'eval_runtime': 97.5961,  'eval_samples_per_second': 16.497, 'eval_steps_per_second': 1.383, 'epoch': 23.0},
    {'eval_loss': 2.0442028045654297, 'eval_rouge1': 32.2368, 'eval_rouge2': 14.7457, 'eval_rougeL': 30.8147, 'eval_rougeLsum': 30.8118, 'eval_gen_len': 16.5404, 'eval_runtime': 99.9381,  'eval_samples_per_second': 16.11,  'eval_steps_per_second': 1.351, 'epoch': 24.0},
    {'eval_loss': 2.0395710468292236, 'eval_rouge1': 31.9379, 'eval_rouge2': 14.5283, 'eval_rougeL': 30.5119, 'eval_rougeLsum': 30.5077, 'eval_gen_len': 16.628,  'eval_runtime': 99.7226,  'eval_samples_per_second': 16.145, 'eval_steps_per_second': 1.354, 'epoch': 25.0},
    {'eval_loss': 2.0357635021209717, 'eval_rouge1': 31.8631, 'eval_rouge2': 14.3336, 'eval_rougeL': 30.4894, 'eval_rougeLsum': 30.5038, 'eval_gen_len': 16.623,  'eval_runtime': 98.562,   'eval_samples_per_second': 16.335, 'eval_steps_per_second': 1.37,  'epoch': 26.0},
    {'eval_loss': 2.0303990840911865, 'eval_rouge1': 32.2028, 'eval_rouge2': 14.5608, 'eval_rougeL': 30.7196, 'eval_rougeLsum': 30.7196, 'eval_gen_len': 16.5975, 'eval_runtime': 99.3551,  'eval_samples_per_second': 16.204, 'eval_steps_per_second': 1.359, 'epoch': 27.0},
    {'eval_loss': 2.032470941543579,  'eval_rouge1': 32.2778, 'eval_rouge2': 14.812,  'eval_rougeL': 30.9333, 'eval_rougeLsum': 30.9061, 'eval_gen_len': 16.5876, 'eval_runtime': 100.2833, 'eval_samples_per_second': 16.055, 'eval_steps_per_second': 1.346, 'epoch': 28.0},
    {'eval_loss': 2.0278966426849365, 'eval_rouge1': 32.4165, 'eval_rouge2': 14.8875, 'eval_rougeL': 31.0581, 'eval_rougeLsum': 31.0613, 'eval_gen_len': 16.564,  'eval_runtime': 99.9576,  'eval_samples_per_second': 16.107, 'eval_steps_per_second': 1.351, 'epoch': 29.0},
    {'eval_loss': 2.0279908180236816, 'eval_rouge1': 32.5664, 'eval_rouge2': 14.7919, 'eval_rougeL': 31.1239, 'eval_rougeLsum': 31.1355, 'eval_gen_len': 16.5814, 'eval_runtime': 103.9718, 'eval_samples_per_second': 15.485, 'eval_steps_per_second': 1.298, 'epoch': 30.0},
    {'eval_loss': 2.0259294509887695, 'eval_rouge1': 32.5867, 'eval_rouge2': 14.988,  'eval_rougeL': 31.1417, 'eval_rougeLsum': 31.1659, 'eval_gen_len': 16.554,  'eval_runtime': 107.9789, 'eval_samples_per_second': 14.91,  'eval_steps_per_second': 1.25,  'epoch': 31.0},
    {'eval_loss': 2.0258138179779053, 'eval_rouge1': 32.3743, 'eval_rouge2': 14.7076, 'eval_rougeL': 31.0024, 'eval_rougeLsum': 30.9985, 'eval_gen_len': 16.5472, 'eval_runtime': 103.4943, 'eval_samples_per_second': 15.556, 'eval_steps_per_second': 1.304, 'epoch': 32.0},
    {'eval_loss': 2.0219216346740723, 'eval_rouge1': 32.6957, 'eval_rouge2': 14.9064, 'eval_rougeL': 31.2029, 'eval_rougeLsum': 31.2063, 'eval_gen_len': 16.5652, 'eval_runtime': 107.6938, 'eval_samples_per_second': 14.95,  'eval_steps_per_second': 1.254, 'epoch': 33.0},
    {'eval_loss': 2.022082567214966,  'eval_rouge1': 32.6421, 'eval_rouge2': 14.9122, 'eval_rougeL': 31.2357, 'eval_rougeLsum': 31.2205, 'eval_gen_len': 16.577,  'eval_runtime': 105.1989, 'eval_samples_per_second': 15.304, 'eval_steps_per_second': 1.283, 'epoch': 34.0},
    {'eval_loss': 2.021690607070923,  'eval_rouge1': 32.5887, 'eval_rouge2': 14.9122, 'eval_rougeL': 31.1608, 'eval_rougeLsum': 31.1739, 'eval_gen_len': 16.5783, 'eval_runtime': 107.0353, 'eval_samples_per_second': 15.042, 'eval_steps_per_second': 1.261, 'epoch': 35.0},
    {'eval_loss': 2.0194954872131348, 'eval_rouge1': 32.779,  'eval_rouge2': 14.908,  'eval_rougeL': 31.3303, 'eval_rougeLsum': 31.3184, 'eval_gen_len': 16.5416, 'eval_runtime': 105.6811, 'eval_samples_per_second': 15.235, 'eval_steps_per_second': 1.277, 'epoch': 36.0},
    {'eval_loss': 2.018709421157837,  'eval_rouge1': 32.6528, 'eval_rouge2': 14.9279, 'eval_rougeL': 31.2705, 'eval_rougeLsum': 31.2803, 'eval_gen_len': 16.5646, 'eval_runtime': 103.1252, 'eval_samples_per_second': 15.612, 'eval_steps_per_second': 1.309, 'epoch': 37.0},
    {'eval_loss': 2.018906831741333,  'eval_rouge1': 32.5526, 'eval_rouge2': 14.8016, 'eval_rougeL': 31.1343, 'eval_rougeLsum': 31.1339, 'eval_gen_len': 16.587,  'eval_runtime': 102.8152, 'eval_samples_per_second': 15.659, 'eval_steps_per_second': 1.313, 'epoch': 38.0},
    {'eval_loss': 2.018338203430176,  'eval_rouge1': 32.6839, 'eval_rouge2': 14.8936, 'eval_rougeL': 31.2802, 'eval_rougeLsum': 31.2723, 'eval_gen_len': 16.5845, 'eval_runtime': 103.8713, 'eval_samples_per_second': 15.5,   'eval_steps_per_second': 1.3,   'epoch': 39.0},
    {'eval_loss': 2.01851749420166,   'eval_rouge1': 32.66,   'eval_rouge2': 14.8576, 'eval_rougeL': 31.2186, 'eval_rougeLsum': 31.2238, 'eval_gen_len': 16.5795, 'eval_runtime': 104.084,  'eval_samples_per_second': 15.468, 'eval_steps_per_second': 1.297, 'epoch': 40.0},
]

train_loss          = [epoch['loss']          for epoch in training_stats]
train_epoch         = [epoch['epoch']         for epoch in training_stats]
train_learning_rate = [epoch['learning_rate'] for epoch in training_stats]

val_loss   = [epoch['eval_loss']   for epoch in validation_stats]
val_epoch  = [epoch['epoch']       for epoch in validation_stats]
val_rouge1 = [epoch['eval_rouge1'] for epoch in validation_stats]
val_rougeL = [epoch['eval_rougeL'] for epoch in validation_stats]

plt.plot(val_epoch, val_rouge1, label='Val_Rouge1')
plt.plot(val_epoch, val_rougeL, label='Val_RougeL')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('ROUGE Score')
plt.savefig('rouge_curve.png', dpi=150, bbox_inches='tight')
plt.show()

"""INFERENCE"""

model_path = "t5-base-finetuned-hinge"

model     = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

device_count = torch.cuda.device_count()
if device_count > 0:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
model.to(device)

prefix = "Generate Hinglish from English: "
input_text = prefix + "what are you doing that u can't see that? Do u realise the reality here?"
input_ids  = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

outputs = model.generate(input_ids, max_new_tokens=150)
print(tokenizer.decode(outputs[0]))