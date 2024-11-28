import fire
import os
from typing import Any
from tqdm import tqdm


from transformers import (
    AutoTokenizer,
    Trainer, 
    TrainingArguments,
)
from transformers import PreTrainedTokenizerFast
from unsloth import FastLanguageModel
from trl import DataCollatorForCompletionOnlyLM
import torch

from dataset import *
from llama_cpp import Llama

tqdm.pandas()

os.environ["WANDB_DISABLED"] = "true"

def load_tokenizer_and_model(model_name: str, max_seq_length: int) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        use_fast=True, 
        padding_side="right"
    )
    model, _ = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
        use_cache=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16, 
        target_modules=[
            "q_proj", 
            "k_proj", 
            "v_proj", 
            "o_proj",
            "gate_proj", 
            "up_proj", 
            "down_proj"
        ],
        lora_alpha=16,
        lora_dropout=0.0, 
        bias="none",   
        use_gradient_checkpointing=True,
        random_state=3407,
        use_rslora=False,  
        loftq_config=None, 
    )
    return tokenizer, model
    
def load_train_dataset_and_collator(
        dataset_path: str,
        tokenizer: PreTrainedTokenizerFast, 
        max_length: int,
) -> Any:

    dataset = DatasetBuilder.get_train_dataset(
        dataset_path=dataset_path, 
        tokenizer=tokenizer, 
        max_length=max_length
    )
    dataset = dataset.shuffle(seed=42)
        
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template="<|im_start|>assistant\n",
        tokenizer=tokenizer
    )
    return dataset, data_collator

def train() -> None:
    max_length = 4096
    tokenizer, model = load_tokenizer_and_model(
        model_name="Qwen/Qwen2.5-14B-Instruct",
        max_seq_length=max_length,
    )
    train_dataset, data_collator = load_train_dataset_and_collator(
        dataset_path="./data/data_training_3103_3110_final_clean.xlsx",
        tokenizer=tokenizer,
        max_length=max_length,
    )
    
    training_args = TrainingArguments(
        output_dir="checkpoint",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=32,
        weight_decay=0.001,
        learning_rate=2e-4,
        max_grad_norm=1.0,
        num_train_epochs=5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        log_level="info",
        logging_strategy="steps",
        logging_steps=5,
        save_strategy="epoch",
        save_only_model=True,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_torch_fused",
        report_to=None,
        group_by_length=True,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    trainer.train()

    if trainer.is_fsdp_enabled:
      trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    output_dir = training_args.output_dir
    model.save_pretrained_gguf(output_dir, tokenizer, quantization_method="q5_k_m")

if __name__ == "__main__":
     fire.Fire(train)