import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from huggingface_hub import login
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os

print("----initializing fsdp--------")
# Initialize distributed training

rank = int(os.getenv('RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))
print(f"Rank: {rank}, World Size: {world_size}")


def cleanup():
    """Clean up the distributed process group."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

def print_multi_gpu_utilization():
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # In GB
        reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)  # In GB
        print(f"GPU {i} memory allocated: {allocated:.2f} GB")
        print(f"GPU {i} memory reserved: {reserved:.2f} GB")\
        

# Check if multiple GPUs are available and initialize process group
if torch.cuda.device_count() > 1:
    torch.distributed.init_process_group(backend='nccl')
else:
    print("Single GPU detected. Skipping torch.distributed initialization.")

# Load the MllamaForConditionalGeneration model and processor
print("----loading model and processor--------")
hf_token = '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'
login(token=hf_token)

model_id = "meta-llama/Llama-3.2-3B"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Add a pad token if it doesn't exist
if tokenizer.pad_token is None:
    # You can use the eos_token as pad_token if there's no specific pad_token
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16  # Use bf16 for better memory efficiency if supported
)


# Move the model to the correct device based on the current rank
device = torch.device(f"cuda:{rank}")
model = model.to(device)
# Now wrap the model in FSDP
model = FSDP(model)

# Check GPU utilization
print_multi_gpu_utilization()

# Prepare the dataset
print("----loading dataset--------")
dataset = load_dataset('cnn_dailymail', '3.0.0')
small_dataset = dataset['train'].select(range(5000))

train_val_split = small_dataset.train_test_split(test_size=0.2)
train_dataset = train_val_split['train']
validation_dataset = train_val_split['test']

def tokenize_function(examples):
    # Tokenize the input article text (inputs)
    inputs = tokenizer(
        examples["article"], 
        return_tensors="pt", 
        truncation=True, 
        padding="max_length", 
        max_length=1024  # or the desired maximum input length
    )
    
    # Tokenize the target highlights (labels)
    labels = tokenizer(
        examples["highlights"], 
        return_tensors="pt", 
        truncation=True, 
        padding="max_length", 
        max_length=1024  # Ensure labels are padded to the same max_length
    )
    
    # Shift the labels for causal language modeling
    labels["input_ids"] = torch.roll(labels["input_ids"], shifts=-1, dims=-1)
    labels["input_ids"][:, -1] = tokenizer.pad_token_id  # Set the last token to padding

    # Add the tokenized labels to the inputs as "labels"
    inputs["labels"] = labels["input_ids"]
    
    return inputs


print("----tokenizing dataset--------")
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = validation_dataset.map(tokenize_function, batched=True)

# Ensure that each process gets its own shard of the data
train_sampler = DistributedSampler(tokenized_train_dataset) if torch.distributed.is_initialized() else None
val_sampler = DistributedSampler(tokenized_val_dataset) if torch.distributed.is_initialized() else None

train_dataloader = DataLoader(
    tokenized_train_dataset,
    sampler=train_sampler,
    batch_size=1,  # Adjust batch size as needed
)

val_dataloader = DataLoader(
    tokenized_val_dataset,
    sampler=val_sampler,
    batch_size=1,  # Adjust batch size as needed
)

torch.cuda.empty_cache()

# Training arguments
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=1,  # Reduce batch size to avoid OOM
    per_device_eval_batch_size=1,
    num_train_epochs=1,  
    gradient_accumulation_steps=8,  # Simulate a larger batch size
    fp16=True,  # Enable mixed precision training
    report_to="none",
    optim="adamw_torch",
    logging_dir="./logs",
    fsdp="full_shard",  # Enables FSDP full-shard strategy
    
    fsdp_config={
        # "transformer_layer_cls_to_wrap": "LlamaDecoderLayer",  # Wrap specific layers only
        "activation_checkpointing": ["LlamaDecoderLayer"]  # Enable activation checkpointing
    },
    ddp_find_unused_parameters=False,  # Important for DDP efficiency
    remove_unused_columns=False,
)




# FSDP Configuration

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
   
)




# Check GPU utilization again before training
print_multi_gpu_utilization()

# Start training
print("----starting training--------")
trainer.train()

# Check GPU utilization after training
print_multi_gpu_utilization()

# Save the model
print("----saving the model--------")
trainer.save_model('./finetuned_llama_11b')

# Clean up distributed resources

cleanup()
