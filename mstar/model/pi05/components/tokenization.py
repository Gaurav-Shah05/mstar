"""Tokenization wrapper for Pi0.5: the PaliGemma tokenizer."""

import torch

from mstar.model.pi05.config import Pi05Config


class Pi05Tokenizer:
    """Wrapper around the HF PaliGemma tokenizer.

    The prompt is encoded as given, with a leading BOS and no other
    normalization, as openpi's pi05 ``PaligemmaTokenizer.tokenize`` does; the
    robot state is already part of it (see ``Pi05Model.process_prompt``).
    """

    def __init__(self, hf_tokenizer, config: Pi05Config):
        self.hf_tokenizer = hf_tokenizer
        self.config = config

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        ids = self.hf_tokenizer(prompt, add_special_tokens=True).input_ids
        ids = ids[: self.config.max_lang_tokens]
        return torch.tensor(ids, dtype=torch.long)
