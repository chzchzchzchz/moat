import json
from typing import List
from tokenizers import Tokenizer

class LlamaTokenizer:
    def __init__(self, tokenizer_path: str):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    @property
    def bos_token_id(self) -> int:
        return 1
        
    @property
    def eos_token_id(self) -> int:
        return 2
        
    @property
    def pad_token_id(self) -> int:
        return 0
