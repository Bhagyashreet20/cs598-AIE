import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from accelerate import Accelerator
from huggingface_hub import login  # Ensure login function is imported
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, BackwardPrefetch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

try:
    # Initialize the Accelerator with FSDP configuration
    accelerator = Accelerator(
        mixed_precision="no",  # No mixed precision as per config
        fsdp_config={
            "sharding_strategy": "FULL_SHARD",
            "offload_params": True,
            "auto_wrap_policy": transformer_auto_wrap_policy,
            "use_no_split_modules": True,
            "backward_prefetch_policy": BackwardPrefetch.BACKWARD_PRE,
            "state_dict_type": StateDictType.SHARDED_STATE_DICT,
            "forward_prefetch": True,
            "use_orig_params": True,
            "activation_checkpointing": True,
            "cpu_offload": True,
        },
    )

    # Login to Hugging Face Hub
    hf_token = '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'
    login(token=hf_token)

    # Load the model and tokenizer
    model_name = "meta-llama/Llama-3.2-3B"  # Using the specific Hugging Face model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Prepare model with Accelerator, then wrap with FSDP
    model = accelerator.prepare(model)
    model = FSDP(model)

    # Load and preprocess the dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        report_to="none",
        gradient_accumulation_steps=1,  # Set accumulation steps if needed
    )

    # Get data loaders and prepare for Accelerate with FSDP
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, collate_fn=data_collator)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=training_args.per_device_eval_batch_size, collate_fn=data_collator)

    # Prepare model, optimizer, and data loaders for Accelerator
    model, train_dataloader, eval_dataloader = accelerator.prepare(model, train_dataloader, eval_dataloader)

    # Optimizer setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)

    # Training loop with Accelerator and FSDP
    for epoch in range(training_args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / training_args.gradient_accumulation_steps

            # Backward pass
            accelerator.backward(loss)
            if step % training_args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        # Evaluation step
        model.eval()
        eval_loss = 0
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model(**batch)
                eval_loss += outputs.loss.item()
        print(f"Epoch {epoch + 1}, Evaluation Loss: {eval_loss / len(eval_dataloader)}")

    # Save the fine-tuned model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

except Exception as e:
    print(f"An error occurred: {e}")
