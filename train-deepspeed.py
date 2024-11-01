import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from huggingface_hub import login
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import deepspeed
import os

# Initialize DeepSpeed and login to Hugging Face Hub
hf_token = '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'
login(token=hf_token)

# Model configuration
model_id = "meta-llama/Llama-3.2-3B"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16  # Use bf16 for better memory efficiency if supported
)

# Initialize DeepSpeed
ds_config = "ds_config.json"

# Prepare the dataset
dataset = load_dataset('cnn_dailymail', '3.0.0')
small_dataset = dataset['train'].select(range(5000))
train_val_split = small_dataset.train_test_split(test_size=0.2)
train_dataset = train_val_split['train']
validation_dataset = train_val_split['test']

# Tokenization
def tokenize_function(examples):
    inputs = tokenizer(
        examples["article"], 
        return_tensors="pt", 
        truncation=True, 
        padding="max_length", 
        max_length=1024
    )
    labels = tokenizer(
        examples["highlights"], 
        return_tensors="pt", 
        truncation=True, 
        padding="max_length", 
        max_length=1024
    )
    labels["input_ids"] = torch.roll(labels["input_ids"], shifts=-1, dims=-1)
    labels["input_ids"][:, -1] = tokenizer.pad_token_id
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = validation_dataset.map(tokenize_function, batched=True)

# DeepSpeed Trainer arguments
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    gradient_accumulation_steps=8,
    fp16=True,
    logging_dir="./logs",
    deepspeed=ds_config,  # Specify the DeepSpeed config file
    report_to="none",
    remove_unused_columns=False,
)

# Initialize Trainer with DeepSpeed
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model('./finetuned_llama_3b')
