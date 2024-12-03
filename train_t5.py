import os
import torch.distributed as dist
from datetime import datetime
from dataclasses import dataclass
import torch
# from torch import 
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForSeq2Seq,
    GPT2LMHeadModel, 
    GPT2Tokenizer
)
from datasets import Dataset
import numpy as np

global rank, device, transfer_group
def init_distributed():
   global rank, device, transfer_group
   rank = int(os.environ["LOCAL_RANK"])
   transfer_ranks = [0,1]
#    world_size = int(os.environ["WORLD_SIZE"])
   device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
   dist.init_process_group(backend="nccl")
   transfer_group = dist.new_group(transfer_ranks)

if __name__ == "__main__":
    rank = int(os.getenv("LOCAL_RANK"))
    print(f"rank: {rank}, init distributed")
    init_distributed()
    model_save_dir = "/projects/bdof/leatherman/t5_checkpoints"
    if rank==0 or rank==1:
        model_name = "t5-small"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name, device_map={"": device})
        
        # inputs = [
        #     "translate English to German: How old are you?",
        #     "translate English to German: Who are you?"
        # ]
        # inputs = tokenizer(inputs, return_tensors="pt", padding=True).to(device)
        # # input_ids = tokenized_inputs['input_ids']  # Extract input IDs

        # outputs = model.generate(input_ids=inputs["input_ids"],
        #                          attention_mask=inputs["attention_mask"],
        #                             do_sample=False, )
        # print(tokenizer.batch_decode(outputs,skip_special_tokens=True))

        # Dummy data
        train_data = {
            'input': [
                "translate English to German: How are you?",
                "translate English to German: What is your name?",
                "translate English to German: Where do you live?",
                "translate English to German: I love programming."
            ],
            'target': [
                "Wie geht es dir?",
                "Wie heißt du?", 
                "Wo wohnst du?",
                "Ich liebe Programmieren."
            ]
        }

        # # # Convert to Hugging Face Dataset
        dataset = Dataset.from_dict(train_data)

        # # Preprocessing function
        def preprocess_function(examples):
            inputs = examples['input']
            targets = examples['target']
            
            model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
            
            # Prepare decoder inputs and labels
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        # Preprocess the dataset
        processed_dataset = dataset.map(
            preprocess_function, 
            batched=True, 
            remove_columns=dataset.column_names
        )

        # # Data collator
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        # # Training arguments
        training_args = TrainingArguments(
            output_dir=model_save_dir,  # Model checkpoint directory
            num_train_epochs=2,
            per_device_train_batch_size=2,
            save_only_model=True,
            # warmup_steps=500,
            # weight_decay=0.01,
            # evaluation_strategy="epoch"
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=processed_dataset,
            data_collator=data_collator,
            transfer_group=transfer_group,
            device = device,
            rank=rank,
            
        )

        # Train the model
        trainer.train()
        print("training finished")
        # Save the final model
        # trainer.save_model(model_save_dir)
    

    