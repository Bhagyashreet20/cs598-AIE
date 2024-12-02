# ... [existing imports] ...
import argparse
import os
import random
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)
from accelerate import Accelerator, DataLoaderConfiguration, DistributedType
from rouge import Rouge  # New import
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from torch.profiler import profile, record_function, ProfilerActivity
import time
import torch.cuda.nvtx as nvtx
import numpy as np

MAX_GPU_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4

model_id = "meta-llama/Llama-3.2-3B"


rank = int(os.environ.get("RANK", 0))  # Global rank
local_rank = int(os.environ.get("LOCAL_RANK", 0))  # Rank on the current node

import time
import torch

def retrieve_fsdp_states(accelerator, model, optimizer):
    # Ensure all GPU operations are complete before timing
    torch.cuda.synchronize()
    start_time = time.time()

    # Unwrap the model and move the state dict to CPU
    model_dl_state_dict = accelerator.unwrap_model(model).state_dict()

    
    model_dl_state_dict_cpu = {k: v.cpu() for k, v in model_dl_state_dict.items()}

    # Measure time for model transfer
    torch.cuda.synchronize()
    model_transfer_time = time.time() - start_time
    print(f"Model state transfer time: {model_transfer_time:.4f} seconds")

    # Start timing optimizer transfer
    start_time = time.time()

    # Move optimizer state to CPU
    optimizer_state = optimizer.state_dict()
    for key in optimizer_state["state"]:
        for state_key, value in optimizer_state["state"][key].items():
            if isinstance(value, torch.Tensor):
                optimizer_state["state"][key][state_key] = value.cpu()

    # Measure time for optimizer transfer
    torch.cuda.synchronize()
    optimizer_transfer_time = time.time() - start_time
    print(f"Optimizer state transfer time: {optimizer_transfer_time:.4f} seconds")

    # Retrieve random states (these are not GPU operations, so no timing needed)
    import random
    import numpy as np
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_random_state_cpu = torch.get_rng_state()
    torch_random_state_gpu = torch.cuda.get_rng_state()

    # Return all states
    return {
        "model_state": model_dl_state_dict_cpu,
        "optimizer_state": optimizer_state,
        "random_states": {
            "python_random": random_state,
            "numpy_random": numpy_state,
            "torch_cpu_random": torch_random_state_cpu,
            "torch_gpu_random": torch_random_state_gpu,
        },
    }




def training_function(config, args):
    # Initialize accelerator
    print("Initializing accelerator")
    print("args.mixed_precision",args.mixed_precision)
    dataloader_config = DataLoaderConfiguration(use_stateful_dataloader=args.use_stateful_dataloader)
    if args.with_tracking:
        accelerator = Accelerator(
            cpu=args.cpu,
            mixed_precision=args.mixed_precision,
            dataloader_config=dataloader_config,
            log_with="all",
            project_dir=args.project_dir,
        )
    else:
        accelerator = Accelerator(
            cpu=args.cpu, mixed_precision=args.mixed_precision, dataloader_config=dataloader_config
        )

    if hasattr(args.checkpointing_steps, "isdigit"):
        if args.checkpointing_steps == "epoch":
            checkpointing_steps = args.checkpointing_steps
        elif args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
        else:
            raise ValueError(
                f"Argument `checkpointing_steps` must be either a number or `epoch`. `{args.checkpointing_steps}` passed."
            )
    else:
        checkpointing_steps = None
    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    lr = config["lr"]
    num_epochs = int(config["num_epochs"])
    seed = int(config["seed"])
    batch_size = int(config["batch_size"])

    # We need to initialize the trackers we use, and also store our configuration
    if args.with_tracking:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run, config)

    print("Preparing dataset")
    datasets = load_dataset("cnn_dailymail", "3.0.0")
    # Split train and validation datasets to get subsets directly
    train_dataset = datasets["train"].train_test_split(train_size=4000, seed=42)["train"]
    val_dataset = datasets["validation"].train_test_split(test_size=1000, seed=42)["test"]

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Define the preprocessing function for summarization
    def preprocess_function(examples):
        inputs = examples["article"]
        targets = examples["highlights"]

        # Tokenize inputs
        model_inputs = tokenizer(inputs, max_length=512, truncation=True)

        # Tokenize targets separately
        labels = tokenizer(targets, max_length=512, truncation=True)
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Add a pad token if it doesn't exist
    if tokenizer.pad_token is None:
        # You can use the eos_token as pad_token if there's no specific pad_token
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    tokenizer.padding_side = "left"

    # Apply the method we just defined to all the examples in all the splits of the dataset
    with accelerator.main_process_first():
        tokenized_train_dataset = train_dataset.map(
            preprocess_function, batched=True, remove_columns=["article", "highlights"]
        )
        tokenized_val_dataset = val_dataset.map(
            preprocess_function, batched=True, remove_columns=["article", "highlights"]
        )

    # If the batch size is too big we use gradient accumulation
    gradient_accumulation_steps = 1
    if batch_size > MAX_GPU_BATCH_SIZE and accelerator.distributed_type != DistributedType.XLA:
        gradient_accumulation_steps = batch_size // MAX_GPU_BATCH_SIZE
        batch_size = MAX_GPU_BATCH_SIZE

    def collate_fn(examples):
        max_length = 512

        inputs = [example["input_ids"] for example in examples]
        labels = [example["labels"] for example in examples]

        model_inputs = tokenizer.pad(
            {"input_ids": inputs},
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        attention_mask = model_inputs["attention_mask"]

        labels = tokenizer.pad(
            {"input_ids": labels},
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
            
        )["input_ids"]

        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        model_inputs["attention_mask"] = attention_mask
        # for key in model_inputs:
        #     if model_inputs[key].dtype not in [torch.float32, torch.float16]:
        #         print("model_inputs[key] is in int64 format",model_inputs[key])
        #         model_inputs[key] = model_inputs[key].float()
      
        return model_inputs

    train_dataloader = DataLoader(
        tokenized_train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=batch_size
    )
    eval_dataloader = DataLoader(
        tokenized_val_dataset, shuffle=False, collate_fn=collate_fn, batch_size=EVAL_BATCH_SIZE
    )

    set_seed(seed)

    # Instantiate the model
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    # Ensure the embedding layer matches the tokenizer vocab size
   
   
 
    model = model.to(accelerator.device)

    # Instantiate optimizer
    optimizer = AdamW(params=model.parameters(), lr=lr)

    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=(len(train_dataloader) * num_epochs) // gradient_accumulation_steps,
    )

    # Prepare everything
    print("Preparing model")
    vocab_size, hidden_size = model.get_input_embeddings().weight.size()

    accelerator.wait_for_everyone()
  
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
  
    # Keep track of steps and epochs
    overall_step = 0
    starting_epoch = 0

    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    # Start training
    print("Start training")
    total_samples = 0
    train_start_time = time.time()
    for epoch in range(starting_epoch, num_epochs):
        epoch_start = time.time()
        # model = fix_flattened_embedding(model, vocab_size=vocab_size, hidden_size=hidden_size)
       
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We need to skip steps until we reach the resumed step
            if not args.use_stateful_dataloader:
                active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            else:
                active_dataloader = train_dataloader
            overall_step += resume_step
        else:
            active_dataloader = train_dataloader

   
                        
            for step, batch in enumerate(active_dataloader):
       
                step_start_time = time.time()
                batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
                loss = outputs.loss
                loss = loss / gradient_accumulation_steps
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if step % gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                
                total_samples += batch["input_ids"].size(0)
                overall_step += 1
                total_step_time = time.time()-step_start_time
                print(f"Step:{step} took {total_step_time:.4f} seconds")
                        
            
                if isinstance(checkpointing_steps, int):
                   
                    output_dir = f"step_{overall_step}"
                    if overall_step % checkpointing_steps == 0:
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        
                        print("---moving the model to cpu manually--")
                        result = retrieve_fsdp_states(accelerator,model,optimizer)
                        # nvtx.range_push(f"Step:{step} Level Checkpointing")
                        # start_ckpt = time.time()
                         
                        # accelerator.save_state(output_dir)
                        # end_ckpt = time.time()
                        # nvtx.range_pop()
                        # print(f"Checkpointing took {end_ckpt - start_ckpt:.4f} seconds")
                        
                        
                        
                            
                            
                                     


            print(f"Epoch:{epoch} took {time.time()-epoch_start:.4f} seconds")
            
            if checkpointing_steps == "epoch": 
                    print("epoch level profiling is turned on!!")
                    output_dir = f"epoch_{epoch}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    start_ckpt = time.time()
                    nvtx.range_push("Epoch Level Checkpointing") 
                    accelerator.save_state(output_dir)
                    end_ckpt = time.time()
                    print(f"Checkpointing took {end_ckpt - start_ckpt:.4f} seconds")
                    nvtx.range_pop()
                 
                    
                            
    print("End training")
    accelerator.end_training()
    total_training_time = time.time() - train_start_time
    print("total_samples in the end",total_samples)
    print("total_training_time",total_training_time)
    throughput = total_samples / total_training_time
    print(f"Throughput: {throughput:.2f} samples/second")

def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--use_stateful_dataloader",
        action="store_true",
        help="If the dataloader should be a resumable stateful dataloader.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to load in all available experiment trackers from the environment and use them for logging.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Optional save directory where all checkpoint folders will be stored. Default is the current working directory.",
    )
    parser.add_argument(
        "--project_dir",
        type=str,
        default="./logs/training-logs",
        help="Location on where to store experiment tracking logs and relevant project information",
    )
    args = parser.parse_args()
    config = {"lr": 1e-5, "num_epochs": 1, "seed": 42, "batch_size": 4}
    training_function(config, args)

if __name__ == "__main__":
    main()
