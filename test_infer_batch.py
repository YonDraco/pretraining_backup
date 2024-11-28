import pandas as pd
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from dataset import Prompter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

prompter = Prompter()

def load_model_and_tokenizer(model_path: str, max_model_len: str,  gpu_memory_utilization: float = 0.95) -> LLM:
    llm = LLM(
        model=model_path, max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization
    )
    return llm

def run(
    tweet: str | list[str],
    model: LLM,
    prompter: Prompter = prompter,
    temperature: float = 0.1,
    max_tokens: int = 128,
    **kwargs
) -> list[str]:

    if isinstance(tweet, str):
        tweet = [tweet]

    messages: list[list[dict[str, str]]] = []
    for tweet_ in tweet:
        message = [
            {"role": "system", "content": prompter.system_prompt},
            {"role": "user", "content": prompter.prompt_template.format(tweet=tweet_)}
        ]
        messages.append(message)

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_k=kwargs.get("top_k", 40),
        top_p=kwargs.get("top_p", 1.0),
        repetition_penalty=kwargs.get("repetition_penalty", 1.0),
        guided_decoding=GuidedDecodingParams(choice=["calling", "not_calling"])
    )

    outputs = model.chat(
        messages=messages, 
        sampling_params=sampling_params,
        use_tqdm=True,
    )

    outputs = [output.outputs[0].text.strip() for output in outputs]
    return outputs

def predict_labels(excel_file: str, model: LLM, batch_size: int= 32) -> pd.DataFrame:
    df = pd.read_excel(excel_file)
    
    if 'full_text' not in df.columns:
        raise ValueError("Input Excel file must contain 'full_text' column.")
    
    tweets = df["full_text"].tolist()
    predicts = []
    for i in range(0, len(tweets), batch_size):
        batch_tweet = tweets[i: i + batch_size]
        batch_predict = run(tweet=batch_tweet, model=model)
        predicts.extend(batch_predict)

    df['predict'] = predicts
    output_file = 'predicted_labels_100sample.xlsx'
    df.to_excel(output_file, index=False)
    print(f"Labels predicted and saved to {output_file}")

    return df

def evaluate(predictions_df: pd.DataFrame):
    if 'action_prompt' not in predictions_df or 'predict' not in predictions_df:
        raise ValueError("Dataframe must contain 'action_prompt' and 'predict' columns for evaluation.")

    true_labels = predictions_df['action_prompt']
    predicted_labels = predictions_df['predict']

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=1)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=1)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=1)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    model_path = "./checkpoint/unsloth.Q5_K_M.gguf"
    max_model_len = 4096
    gpu_memory_utilization = 0.4
    batch_size = 512
    model = load_model_and_tokenizer(
        model_path=model_path,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    excel_file = "./data/data_infer_100sample.xlsx"
    
    predictions_df = predict_labels(excel_file, model, batch_size)
    
    evaluate(predictions_df)