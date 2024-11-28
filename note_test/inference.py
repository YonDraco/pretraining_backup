import pandas as pd
from unsloth import FastLanguageModel
from dataset import Prompter
import torch

prompter = Prompter

def load_model_and_tokenizer(
    model_name_or_path: str,
    load_in_4bit: bool = True,
    max_seq_length: int = 4096,
):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_or_path, 
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def run(
    tweet: str,
    model, 
    tokenizer,
    prompter: Prompter = prompter, 
    device: int | str = "cuda:0",
    temperature: float = 0.5,
    top_k: int = 40,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    use_cache: bool = True,
    **kwargs
) -> str:
    
    conversation = [
        {"role": "system", "content": prompter.system_prompt},
        {"role": "user", "content": prompter.prompt_template.format(tweet=tweet)}
    ]
    input_ids = tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            use_cache=use_cache,
            **kwargs
        )
    
    output_text = tokenizer.batch_decode(outputs[:, input_ids.size(1): ], skip_special_tokens=True)[0].strip()
    torch.cuda.empty_cache()

    return output_text

def predict_labels(excel_file: str, model, tokenizer):
    df = pd.read_excel(excel_file)

    df['predict'] = df['full_text'].apply(lambda tweet: run(tweet, model, tokenizer))

    output_file = 'predicted_labels_non_imb.xlsx'  
    df.to_excel(output_file, index=False)

    print(f"Labels predicted and saved to {output_file}")

if __name__ == "__main__":
    model_name_or_path = "./checkpoint/checkpoint-400"
    load_in_4bit = False
    max_seq_length = 4096

    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        load_in_4bit=load_in_4bit,
        max_seq_length=max_seq_length
    )

    excel_file = "./data/data_infer_100sample.xlsx"  
    predict_labels(excel_file, model, tokenizer)
