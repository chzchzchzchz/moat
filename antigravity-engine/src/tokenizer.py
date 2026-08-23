import json
from typing import List
from tokenizers import Tokenizer

class LlamaTokenizer:
    def __init__(self, tokenizer_path: str):
        import os
        model_dir = os.path.dirname(os.path.abspath(tokenizer_path))
        try:
            from transformers import AutoTokenizer
            self.auto_tok = AutoTokenizer.from_pretrained(model_dir)
            self.use_auto = True
        except Exception:
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            self.use_auto = False

    def encode(self, text: str) -> List[int]:
        if self.use_auto:
            return self.auto_tok.encode(text, add_special_tokens=False)
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids: List[int]) -> str:
        if self.use_auto:
            return self.auto_tok.decode(token_ids)
        return self.tokenizer.decode(token_ids)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    @property
    def bos_token_id(self) -> int:
        if self.use_auto and hasattr(self.auto_tok, 'bos_token_id') and self.auto_tok.bos_token_id is not None:
            return self.auto_tok.bos_token_id
        return 151644 if self.vocab_size > 32000 else 1
        
    @property
    def eos_token_id(self) -> int:
        if self.use_auto and hasattr(self.auto_tok, 'eos_token_id') and self.auto_tok.eos_token_id is not None:
            return self.auto_tok.eos_token_id
        return 151645 if self.vocab_size > 32000 else 2
        
    @property
    def pad_token_id(self) -> int:
        if self.use_auto and hasattr(self.auto_tok, 'pad_token_id') and self.auto_tok.pad_token_id is not None:
            return self.auto_tok.pad_token_id
        return 151643 if self.vocab_size > 32000 else 0
