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
import torch.distributed.checkpoint as DCP
import torch.distributed as dist
from torch.distributed.checkpoint import StorageWriter
MAX_GPU_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2

model_id = "meta-llama/Llama-3.2-3B"


rank = int(os.environ.get("RANK", 0))  # Global rank
local_rank = int(os.environ.get("LOCAL_RANK", 0))  # Rank on the current node

# # Initialize the NCCL backend for GPU communication
# dist.init_process_group(backend="nccl", init_method="env://")

# # Initialize the Gloo backend for CPU communication
# dist.init_process_group(backend="gloo", init_method="env://")
# print(f"Backend for GPU: {dist.get_backend()}")



def training_function(config, args):
    # Initialize accelerator
    print("Initializing accelerator")
    print("args.mixed_precision",args.mixed_precision)
    dist.init_process_group(backend="gloo", init_method="env://")

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
            cpu=args.cpu, mixed_precision=args.mixed_precision
            , dataloader_config=dataloader_config
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
    train_dataset = datasets["train"].train_test_split(train_size=100, seed=42)["train"]
    val_dataset = datasets["validation"].train_test_split(test_size=25, seed=42)["test"]

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

    tokenizer = AutoTokenizer.from_pretrained(model_id, torch_dtype=torch.bfloat16)
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
    print("batch_size before if",batch_size)
    if batch_size > MAX_GPU_BATCH_SIZE and accelerator.distributed_type != DistributedType.XLA:
        #Gradients will be accumulated for gradient_accumulation_steps micro-batches before updating the model parameters.
        gradient_accumulation_steps = batch_size // MAX_GPU_BATCH_SIZE
        batch_size = MAX_GPU_BATCH_SIZE
        print("gradient_accumulation_steps",gradient_accumulation_steps,"batch_size",batch_size)

    print("batch_size",batch_size)
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
      
        return model_inputs

    train_dataloader = DataLoader(
        tokenized_train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=batch_size,drop_last=True
    )
    eval_dataloader = DataLoader(
        tokenized_val_dataset, shuffle=False, collate_fn=collate_fn, batch_size=EVAL_BATCH_SIZE,drop_last=False
    )

    set_seed(seed)

    # Instantiate the model
    model = AutoModelForCausalLM.from_pretrained(model_id)
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
    print("num_traiing steps",(len(train_dataloader) * num_epochs) // gradient_accumulation_steps,"trainloader len",len(train_dataloader))
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
    # Keep track of async checkpoint tasks
    pending_checkpoints = []
    checkpoint_times = []
    for epoch in range(starting_epoch, num_epochs):
        
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

   
            checkpoint_future = None       
            for step, batch in enumerate(active_dataloader):
                   
           
                print("step",step)
                print(f"[Rank {rank}] Step {step}: Input IDs shape {batch['input_ids'].shape}, Labels shape {batch['labels'].shape}")
                batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                assert batch["input_ids"].is_cuda, f"Rank {rank}: Input IDs are not on GPU!"
                print(f"[Rank {rank}] Step {step}: Moved to device - Input IDs shape {batch['input_ids'].shape}, Labels shape {batch['labels'].shape}")

                outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
                loss = outputs.loss
                loss = loss / gradient_accumulation_steps
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                print('backward computation done')
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if step % gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                
                total_samples += batch["input_ids"].size(0)
                overall_step += 1

                        
                # if overall_step % checkpointing_steps == 0:
                #     print("--profiling results--")   
                #     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
                    
                if isinstance(checkpointing_steps, int):
                    
                    print(f"[Rank {dist.get_rank()}] Device: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}")
                    if checkpoint_future is not None:
                        print("completing the existing checkpoint offloading.....")
                        checkpoint_future.result()
                    output_dir = f"async_step_{overall_step}"
                    if overall_step % checkpointing_steps == 0:
                        print(f"checkpointing is happening at step:{overall_step}")
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        os.makedirs(output_dir, exist_ok=True)
                        checkpoint_start_time = time.time()
                        fs_storage_writer = DCP.FileSystemWriter(output_dir)
                       
                       
                        checkpoint_future = DCP.async_save(
                                state_dict={
                                        "model": model.state_dict(),
                                        "optimizer": optimizer.state_dict(),
                                        "lr_scheduler": lr_scheduler.state_dict(),
                                        "epoch": epoch,
                                        "step": step,
                                        
                            },
                            storage_writer=fs_storage_writer,
                           

                        )
                        checkpoint_initiation_time = time.time() - checkpoint_start_time

                       # pending_checkpoints.append(checkpoint_future)
                        checkpoint_times.append({"step": overall_step, "initiation_time": checkpoint_initiation_time})
                        print(f"Async checkpointing initiated for step {overall_step} in {checkpoint_initiation_time:.4f} seconds")
                        
                    
        
            if checkpointing_steps == "epoch":
                    print("epoch level profiling is turned on!!")
            
                    output_dir = f"async_epoch_{epoch}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    checkpoint_start_time = time.time()
                    fs_storage_writer = torch.distributed.checkpoint.FileSystemWriter("/checkpoints")
                    torch.cuda.synchronize()
                    checkpoint_future= DCP.async_save(
                        {
                            "model": model.state_dict(),  
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "epoch": epoch,
                            "step": step,  
                        },
                        
                        storage_writer=fs_storage_writer,
                        
                    )
                    checkpoint_initiation_time = time.time() - checkpoint_start_time
                    pending_checkpoints.append(checkpoint_future)
                    checkpoint_times.append({"step": overall_step, "initiation_time": checkpoint_initiation_time})
                    print(f"Async checkpointing initiated for step {epoch} in {checkpoint_initiation_time:.4f} seconds")
            dist.destroy_process_group()      

    print("All asynchronous checkpoints completed.")                       
    print("End training")
    accelerator.end_training()
    total_training_time = time.time() - train_start_time
    print("total_samples in the end",total_samples)
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
    #TODO: change batch size = 4 when gpu=2, batch=6 when gpu=3
    config = {"lr": 1e-5, "num_epochs": 1, "seed": 42, "batch_size": 6}
    training_function(config, args)

if __name__ == "__main__":
    print("entering main")
    main()
