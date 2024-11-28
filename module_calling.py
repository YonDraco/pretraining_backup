import asyncio
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from typing import Any

from datasets import Dataset, disable_caching
from dataclasses import dataclass
from transformers import PreTrainedTokenizerFast

@dataclass
class Prompter:
    system_prompt: str = (
        "You are an assistant in cryptocurrency markets, blockchain technology, and trading. " 
        "Your task is to analyze the action of tweets related to these topics.\n"
        "For each tweet, determine whether the action is 'calling' or 'not_calling'.\n"
        "'Calling' tweets mention tokens and related buying or selling actions. " 
        "They often contain words like: blanks, push, added, adding, all-time highs, bear trap, bearish is bad, bet, betting, bid, bids, bottom, bottom in, bought, bounce, break out, bullish, buy, buying, call, capitalize, dumping, enter, fading, for buying imo, gift, go long, god, going higher, grab, grind up, higher, highs incoming, just buy, loading, long, longed, longing, lower, moon, more expensive, much higher, price discovery, pump, rally, re-longing, reaccumulation, ride, uptrend, up, sell, upside, send it, short, send it, squeeze up, some longs, sell-off, relief.\n"
        "If the tweet does not fall into the above case, then classify it as 'not_calling'."
    )
    prompt_template: str = "Classify the following Tweet as 'calling' or 'not_calling': {tweet}"

prompter = Prompter

    
class DatasetBuilder:
    
    disable_caching()
    
    def __init__(
        self,
        dataset_path: str,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 64,
        prompter: Prompter = prompter,
    ):
        self.tokenizer = tokenizer
        self.prompter = prompter
        self.max_length = max_length
        self.dataset_path = dataset_path
    
    @property
    def dataset(self) -> Dataset:
        dataset = self._load_dataset(path=self.dataset_path)
        dataset = dataset.map(
            self._dataprocessing,
            remove_columns=dataset.column_names,
            num_proc=8
        )
        return dataset

    @classmethod
    def get_train_dataset(cls, dataset_path: str, tokenizer: PreTrainedTokenizerFast, max_length: int = 64) -> Dataset:
        return cls(dataset_path=dataset_path, tokenizer=tokenizer, max_length=max_length).dataset

    @classmethod
    def get_eval_dataset(cls, eval_dataset_path: str, tokenizer: PreTrainedTokenizerFast, max_length: int = 64) -> Dataset:
        eval_builder = cls(dataset_path=eval_dataset_path, tokenizer=tokenizer, max_length=max_length)
        return eval_builder.dataset

    def _load_dataset(self, path: str) -> Dataset:
        df = pd.read_excel(path)
        dataset = Dataset.from_pandas(df)
        return dataset
    
    def _process_long_text(self, input_ids: list[int]) -> list[int]:
        if len(input_ids) > self.max_length:
            im_end_id = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
            end_inst_indices = input_ids.index(im_end_id, input_ids.index(im_end_id) + 1)
            num_trunc_tokens = len(input_ids) - self.max_length
            if end_inst_indices - num_trunc_tokens <= 0:
                input_ids = input_ids[end_inst_indices:]
            else:
                input_ids = input_ids[:end_inst_indices - num_trunc_tokens] + input_ids[end_inst_indices:]
            return input_ids
        
        return input_ids

    def _dataprocessing(self, sample: dict[str, str]) -> Any:
        tweet = sample["full_text"]
        action = sample["action_prompt"]

        conversation = [
            {"role": "system", "content": self.prompter.system_prompt},
            {"role": "user", "content": self.prompter.prompt_template.format(tweet=tweet)},
            {"role": "assistant", "content": action}
        ]

        input_ids = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            truncation=True,
            max_length=self.max_length,
            add_generation_prompt=False,
        )

        input_ids = self._process_long_text(input_ids=input_ids)
        return dict(input_ids=input_ids)

def load_model_and_tokenizer(model_path: str, max_model_len: str, gpu_memory_utilization: float = 0.95) -> LLM:
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

# async def predict_labels_from_db(model: LLM, batch_size: int = 32) -> pd.DataFrame:
#     async def get_data_tweets():
#         page_size = 10
#         current_page = 1
#         records = []
#         total_rows = 0

#         while True:
#             final_df_processed, total_rows = await mongo_helper.load_tweet_analysis_newest_run_id(
#                 'tweets_analysis_result',
#                 1,
#                 current_page,
#                 page_size
#             )

#             result = final_df_processed.to_dict(orient='records')
#             records.extend(result)

#             if len(records) >= total_rows:
#                 break
#             current_page += 1

#         response = {
#             "totalRow": total_rows,
#             "records": [
#                 {
#                     "id": record["id"],
#                     "type": 1,
#                     "create_at_text": record.get("create_at_text"),
#                     "create_at": pd.to_datetime(record.get("create_at_text")).timestamp() * 1000 if record.get(
#                         "create_at_text") else None,
#                     "post_url": record.get("post_url"),
#                     "full_text": record.get("full_text"),
#                     "screen_name": record.get("screen_name"),
#                     "user_id": record.get("user_id"),
#                     "lang": record.get("lang"),
#                     "detected_coins": record.get("detected_coins"),
#                     "action_prompt": record.get("action_prompt"),
#                     "sentiment_prompt": record.get("sentiment_prompt"),
#                     "term_classification": record.get("term_classification")
#                 }
#                 for record in records
#             ]
#         }
#         return response

#     result = await get_data_tweets()
#     df = pd.DataFrame(result['records'])

#     if 'full_text' not in df.columns:
#         raise ValueError("Fetched data must contain a 'full_text' column.")

#     tweets = df["full_text"].tolist()
#     predicts = []
    
#     for i in range(0, len(tweets), batch_size):
#         batch_tweet = tweets[i: i + batch_size]
#         batch_predict = run(tweet=batch_tweet, model=model)
#         predicts.extend(batch_predict)

#     df['predict'] = predicts
#     print("Labels predicted and added to DataFrame.")
#     return df


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

    df['predict_calling_qwen'] = predicts
    output_file = 'predicted_labels_from_db.xlsx'
    df.to_excel(output_file, index=False)
    print(f"Labels predicted and saved to {output_file}")

    return df


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

    excel_file = "./data/data_backtest_3103_2011.xlsx"
    
    predictions_df = predict_labels(excel_file, model, batch_size)
