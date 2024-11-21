import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from huggingface_hub import login
from datasets import load_dataset
import deepspeed
import os

# Login to Hugging Face Hub
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
    torch_dtype=torch.bfloat16
)
model.gradient_checkpointing_enable()

# Initialize DeepSpeed
ds_config = "ds_config.json"

# Prepare the dataset 
dataset = load_dataset('cnn_dailymail', '3.0.0')
small_dataset = dataset['train'].select(range(5000))
train_val_split = small_dataset.train_test_split(test_size=0.1)
train_dataset = train_val_split['train']
validation_dataset = train_val_split['test']

# Tokenization
def tokenize_function(examples):
    inputs = tokenizer(
        examples["article"],
        return_tensors="pt", 
        truncation=True, 
        padding="max_length", 
        max_length=512
    )
    labels = tokenizer(
        examples["highlights"],
        return_tensors="pt", 
        truncation=True, 
        padding="max_length", 
        max_length=512
    )
    labels["input_ids"] = torch.roll(labels["input_ids"], shifts=-1, dims=-1)
    labels["input_ids"][:, -1] = tokenizer.pad_token_id
    
    # Convert tensors to lists for compatibility with datasets.map
    return {
        "input_ids": inputs["input_ids"].tolist(),
        "attention_mask": inputs["attention_mask"].tolist(),
        "labels": labels["input_ids"].tolist(),
    }

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = validation_dataset.map(tokenize_function, batched=True)

# Training arguments with DeepSpeed
training_args = TrainingArguments(
    output_dir="/projects/bdof/nkanamarla/models",
    num_train_epochs=1,
    save_steps=1,  # Save a checkpoint at every step
    fp16=True,
    logging_dir="/projects/bdof/nkanamarla/deepspeed-logs",
    deepspeed=ds_config,  # Use DeepSpeed config
    report_to="none",
    remove_unused_columns=False,
    save_strategy="epoch",
    evaluation_strategy="epoch"
)

# Initialize Trainer with DeepSpeed
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
)

# Training loop with checkpointing at each epoch
for epoch in range(training_args.num_train_epochs):
    print(f"Starting epoch {epoch + 1}")
    trainer.train()
    print(f"Finished epoch {epoch + 1}")

