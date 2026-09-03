from __future__ import annotations
from collections import Counter
import re


class Vocabulary:
    SPECIAL = ("<pad>", "<bos>", "<eos>", "<unk>")

    def __init__(self, captions: list[str], minimum_frequency: int = 2):
        counts = Counter(token for caption in captions for token in self.tokenize(caption))
        words = sorted(word for word, count in counts.items() if count >= minimum_frequency)
        self.itos = list(self.SPECIAL) + words
        self.stoi = {word: index for index, word in enumerate(self.itos)}

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9']+", text.lower())

    def encode(self, caption: str) -> list[int]:
        unk = self.stoi["<unk>"]
        return [self.stoi["<bos>"]] + [self.stoi.get(token, unk) for token in self.tokenize(caption)] + [self.stoi["<eos>"]]

    def decode(self, indices: list[int]) -> str:
        ignored = {self.stoi["<pad>"], self.stoi["<bos>"]}
        words = []
        for index in indices:
            if index == self.stoi["<eos>"]:
                break
            if index not in ignored:
                words.append(self.itos[index] if 0 <= index < len(self.itos) else "<unk>")
        return " ".join(words)

