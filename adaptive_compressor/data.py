"""Byte-level dataset helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

try:
    from datasets import load_dataset
except ImportError as exc:  # pragma: no cover - import guard for better runtime error
    load_dataset = None
    _DATASETS_IMPORT_ERROR = exc
else:
    _DATASETS_IMPORT_ERROR = None


@dataclass
class ByteDatasetConfig:
    dataset_name: str = "Salesforce/wikitext"
    dataset_config_name: str = "wikitext-103-raw-v1"
    split: str = "train"
    text_field: str = "text"
    sequence_length: int = 128
    max_documents: int | None = 5000


class ByteChunkDataset(Dataset[torch.Tensor]):
    """Fixed-length UTF-8 byte windows cut from a text corpus."""

    def __init__(self, chunks: torch.Tensor) -> None:
        self.chunks = chunks

    def __len__(self) -> int:
        return int(self.chunks.size(0))

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.chunks[index]


def load_wikitext_byte_dataset(config: ByteDatasetConfig) -> ByteChunkDataset:
    """Load WikiText and expose it as fixed-length byte sequences."""

    if load_dataset is None:
        raise ImportError(
            "The 'datasets' package is required to load WikiText. "
            "Install it with `pip install datasets`."
        ) from _DATASETS_IMPORT_ERROR

    dataset = load_dataset(
        config.dataset_name,
        config.dataset_config_name,
        split=config.split,
    )
    if config.max_documents is not None:
        dataset = dataset.select(range(min(len(dataset), config.max_documents)))

    texts = [text for text in dataset[config.text_field] if text]
    merged_text = "\n\n".join(texts)
    byte_values = torch.tensor(list(merged_text.encode("utf-8")), dtype=torch.long)

    usable_tokens = (
        byte_values.numel() // config.sequence_length
    ) * config.sequence_length
    byte_values = byte_values[:usable_tokens]
    chunks = byte_values.view(-1, config.sequence_length)
    return ByteChunkDataset(chunks)
