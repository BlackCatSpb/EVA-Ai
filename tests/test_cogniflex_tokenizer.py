import os
import pytest
import pytest_asyncio

from eva_ai.mlearning.cogniflex_tokenizer import ЕВАTokenizer, TokenizationConfig


def is_offline_env() -> bool:
    return os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"


@pytest.mark.skipif(
    os.environ.get("COGNIFLEX_LOCAL_MODEL_DIR") is None,
    reason="COGNIFLEX_LOCAL_MODEL_DIR не задан, пропуск локального офлайн-теста",
)
@pytest.mark.asyncio
async def test_tokenizer_offline_local_load():
    local_dir = os.environ["COGNIFLEX_LOCAL_MODEL_DIR"]
    assert os.path.isdir(local_dir), "Локальный каталог модели не существует"

    tok = await ЕВАTokenizer.from_pretrained(local_dir, local_files_only=True, use_fast=True)

    assert tok is not None
    assert getattr(tok, "tokenizer", None) is not None

    cfg = TokenizationConfig(max_length=32, padding=True, truncation=True)
    sample = "Привет, как дела?"

    enc = tok.encode(sample, cfg)
    assert isinstance(enc, dict)
    assert "input_ids" in enc and "attention_mask" in enc
    assert enc["input_ids"].shape[0] == 1

    decoded = tok.decode(enc["input_ids"])  # type: ignore[arg-type]
    assert isinstance(decoded, str)
    assert len(decoded) > 0


@pytest.mark.skipif(
    is_offline_env(),
    reason="Офлайн-режим активирован (HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE)",
)
@pytest.mark.asyncio
async def test_tokenizer_online_rugpt3_load():
    model_id = os.environ.get("COGNIFLEX_TEST_MODEL", "sberbank-ai/rugpt3large_based_on_gpt2")

    tok = await ЕВАTokenizer.from_pretrained(model_id, local_files_only=False, use_fast=True)

    assert tok is not None
    assert getattr(tok, "tokenizer", None) is not None

    cfg = TokenizationConfig(max_length=32, padding=True, truncation=True)
    sample = "Это небольшой тест токенизатора ruGPT3."

    enc = tok.encode(sample, cfg)
    assert isinstance(enc, dict)
    assert "input_ids" in enc and "attention_mask" in enc

    decoded = tok.decode(enc["input_ids"])  # type: ignore[arg-type]
    assert isinstance(decoded, str)
    assert len(decoded) > 0

    inner = tok.tokenizer
    pad = getattr(inner, "pad_token", None)
    eos = getattr(inner, "eos_token", None)
    assert pad is not None or eos is not None
