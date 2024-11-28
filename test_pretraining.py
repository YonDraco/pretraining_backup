import os
import fire
from typing import Any
from tqdm import tqdm
import pandas as pd

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import torch

os.environ["WANDB_DISABLED"] = "true"

tqdm.pandas()

def load_tokenizer_and_model(model_name: str, max_seq_length: int) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        use_fast=True, 
        padding_side="right"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    return tokenizer, model

def load_pretrain_dataset(
        dataset_path: str,
        tokenizer: Any,
        max_length: int,
) -> Any:
    if dataset_path.endswith(".xlsx"):
        df = pd.read_excel(dataset_path)
    elif dataset_path.endswith(".csv"):
        df = pd.read_csv(dataset_path)
    else:
        raise ValueError("Dataset must be in .csv or .xlsx format.")

    df["text"] = df["title"].fillna("") + "\n\n" + df["content"].fillna("")
    
    dataset = Dataset.from_pandas(df[["text"]])
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=max_length, 
            padding="max_length"
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized_dataset

def train() -> None:
    max_length = 1024 
    model_name = "Qwen/Qwen2.5-14B-Instruct"

    tokenizer, model = load_tokenizer_and_model(model_name, max_seq_length=max_length)
    
    train_dataset = load_pretrain_dataset(
        dataset_path="./data/data_training_news_example.xlsx",
        tokenizer=tokenizer,
        max_length=max_length,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False  
    )

    training_args = TrainingArguments(
        output_dir="checkpoint",
        overwrite_output_dir=True,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        evaluation_strategy="no",
        save_strategy="epoch",
        save_total_limit=3,
        num_train_epochs=3,
        learning_rate=5e-5,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=50,
        logging_dir="./logs",
        report_to=None,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        group_by_length=True,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    fire.Fire(train)