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

MAX_GPU_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4

model_id = "meta-llama/Llama-3.2-3B"

def training_function(config, args):
    # Initialize accelerator
    print("Initializing accelerator")
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
    train_dataset = datasets["train"].train_test_split(train_size=100, seed=42)["train"]
    val_dataset = datasets["validation"].train_test_split(test_size=100, seed=42)["test"]

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Define the preprocessing function for summarization
    def preprocess_function(examples):
        inputs = examples["article"]
        targets = examples["highlights"]

        # Tokenize inputs
        model_inputs = tokenizer(inputs, max_length=512, truncation=True)

        # Tokenize targets separately
        labels = tokenizer(targets, max_length=128, truncation=True)
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
        # Uncomment the following lines for debugging
        # print(f"Input IDs shape: {model_inputs['input_ids'].shape}")
        # print(f"Labels shape: {model_inputs['labels'].shape}")
        # print(f"Attention mask shape: {model_inputs['attention_mask'].shape}")

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
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Initial model embedding size: {model.get_input_embeddings().weight.size(0)}")
    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized model embedding size: {model.get_input_embeddings().weight.size(0)}")

    print(f"Model vocab size: {model.get_input_embeddings().weight.size(0)}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    assert model.get_input_embeddings().weight.size(0) == len(tokenizer), "Embedding size mismatch!"

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
    for epoch in range(starting_epoch, num_epochs):
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
            # Uncomment the following line for debugging
            # print(batch["input_ids"].shape, batch["labels"].shape)
            # Forward pass with only input_ids and labels for LLaMA
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
            loss = outputs.loss
            loss = loss / gradient_accumulation_steps
            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach().float()
            accelerator.backward(loss)
            if step % gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            overall_step += 1

            if isinstance(checkpointing_steps, int):
                output_dir = f"step_{overall_step}"
                if overall_step % checkpointing_steps == 0:
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

        # Evaluation
        model.eval()
        decoded_predictions = []
        decoded_references = []

        # Collect predictions and references
        for step, batch in enumerate(eval_dataloader):
            # Move batch to device
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_new_tokens=128,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                )

            # Decode predictions
            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(
                torch.where(batch["labels"] != -100, batch["labels"], tokenizer.pad_token_id),
                skip_special_tokens=True,
            )

            decoded_predictions.extend(decoded_preds)
            decoded_references.extend(decoded_labels)

        # Compute ROUGE scores using the `rouge` package
        if accelerator.is_main_process:
            rouge = Rouge()
            scores = rouge.get_scores(decoded_predictions, decoded_references, avg=True)

            # Print or log the results
            print(f"ROUGE scores at epoch {epoch}:")
            for metric, results in scores.items():
                print(f"{metric}:")
                for sub_metric in ['p', 'r', 'f']:
                    print(f"  {sub_metric}: {results[sub_metric]:.4f}")
            if args.with_tracking:
                # Prepare results for logging
                rouge_dict = {f"{metric}_{sub_metric}": results[sub_metric]
                              for metric, results in scores.items()
                              for sub_metric in ['p', 'r', 'f']}
                accelerator.log({"rouge": rouge_dict}, step=overall_step)

            # Optionally, print a few examples
            for pred, ref in zip(decoded_predictions[:5], decoded_references[:5]):
                print(f"Prediction: {pred}")
                print(f"Reference: {ref}")
                print("-" * 80)

        accelerator.print(f"Finished evaluation for epoch {epoch}.")

        if checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    print("End training")
    accelerator.end_training()

def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
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
        default="logs",
        help="Location on where to store experiment tracking logs and relevant project information",
    )
    args = parser.parse_args()
    config = {"lr": 2e-5, "num_epochs": 1, "seed": 42, "batch_size": 4}
    training_function(config, args)

if __name__ == "__main__":
    main()
