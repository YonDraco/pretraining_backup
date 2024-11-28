from typing import Any

from datasets import Dataset, disable_caching
import pandas as pd
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