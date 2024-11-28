import fire
import os
from typing import Any, Dict
from copy import deepcopy

from transformers import (
    AutoTokenizer,
    Trainer, 
    TrainingArguments,
    TrainerCallback
)
from unsloth import FastLanguageModel
from trl import DataCollatorForCompletionOnlyLM
import torch

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch.nn import CrossEntropyLoss

from dataset import *

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
        tokenizer: AutoTokenizer, 
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

def load_eval_dataset(
        eval_dataset_path: str,
        tokenizer: AutoTokenizer,
        max_length: int,
) -> Any:
    eval_dataset = DatasetBuilder.get_eval_dataset(
        eval_dataset_path=eval_dataset_path,
        tokenizer=tokenizer,
        max_length=max_length
    )
    return eval_dataset

def compute_metrics(pred) -> Dict[str, float]:
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    
    loss_fct = CrossEntropyLoss()
    logits = torch.tensor(pred.predictions)
    labels_tensor = torch.tensor(labels)
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels_tensor.view(-1))
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'loss': loss.item(),
    }

class CustomCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        control_copy = deepcopy(control)
        
        if control.should_evaluate:
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
        
        return control_copy

def train() -> None:
    max_length = 4096
    tokenizer, model = load_tokenizer_and_model(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        max_seq_length=max_length,
    )
    train_dataset, data_collator = load_train_dataset_and_collator(
        dataset_path="./data/data_training_calling.xlsx",
        tokenizer=tokenizer,
        max_length=max_length,
    )
    eval_dataset = load_eval_dataset(
        eval_dataset_path="./data/data_infer_100sample.xlsx",
        tokenizer=tokenizer,
        max_length=max_length,
    )
    
    training_args = TrainingArguments(
        output_dir="checkpoint",
        per_device_train_batch_size=2, #4
        gradient_accumulation_steps=8, #32
        weight_decay=0.001,
        learning_rate=2e-4,
        max_grad_norm=1.0,
        num_train_epochs=1,
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
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )
    trainer.add_callback(CustomCallback())

    torch.cuda.empty_cache()
    
    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    
    torch.cuda.empty_cache()
    eval_result = trainer.evaluate()
    print(eval_result)
    trainer.log_metrics("eval", eval_result)
    trainer.save_metrics("eval", eval_result)
    trainer.save_state()

if __name__ == "__main__":
    fire.Fire(train)
