#!/usr/bin/env python3
import os
import json
import shutil
import types

import pytest


@pytest.fixture()
def temp_models_dir(tmp_path):
    d = tmp_path / "models"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def _touch(path: str, content: str = ""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _make_minimal_hf_model(dirpath: str, tokenizer_variant: str = "json"):
    # minimal config
    _touch(os.path.join(dirpath, "config.json"), json.dumps({"architectures": ["GPT2LMHeadModel"]}))
    # weights (choose one)
    _touch(os.path.join(dirpath, "pytorch_model.bin"), "BIN_PLACEHOLDER")
    # tokenizer variants
    if tokenizer_variant == "json":
        _touch(os.path.join(dirpath, "tokenizer.json"), json.dumps({"version": 1}))
    elif tokenizer_variant == "gpt2_merges_vocab":
        _touch(os.path.join(dirpath, "merges.txt"), "# merges")
        _touch(os.path.join(dirpath, "vocab.json"), json.dumps({"hello": 0}))
    else:
        raise ValueError("unknown tokenizer_variant")


def _new_manager(temp_models_dir: str):
    # Import here to ensure test-time module state is fresh
    from eva_ai.mlearning.model_manager import ModelManager
    # safe_test_mode=True prevents heavy init; autoload/background disabled
    mm = ModelManager(brain=None, model_dir=temp_models_dir, safe_test_mode=True)
    return mm


def test_has_local_model_variants(temp_models_dir):
    from eva_ai.mlearning.model_manager import ModelManager

    mm = _new_manager(temp_models_dir)

    tdir = os.path.join(temp_models_dir, "rugpt3_large")
    os.makedirs(tdir, exist_ok=True)

    # 1) empty -> False
    assert mm._has_local_model(tdir) is False

    # 2) minimal with tokenizer.json -> True
    _make_minimal_hf_model(tdir, tokenizer_variant="json")
    assert mm._has_local_model(tdir) is True

    # 3) remove tokenizer.json; use merges+vocab -> True
    try:
        os.remove(os.path.join(tdir, "tokenizer.json"))
    except FileNotFoundError:
        pass
    _touch(os.path.join(tdir, "merges.txt"), "# merges")
    _touch(os.path.join(tdir, "vocab.json"), json.dumps({"hello": 0}))
    assert mm._has_local_model(tdir) is True


def test_provision_skips_download_when_local_exists(monkeypatch, temp_models_dir):
    # Prepare local model directory with minimal files
    local_name = "rugpt3_large"
    target_dir = os.path.join(temp_models_dir, local_name)
    os.makedirs(target_dir, exist_ok=True)
    _make_minimal_hf_model(target_dir, tokenizer_variant="json")

    # Build manager in safe mode to avoid auto-provisioning in __init__
    from eva_ai.mlearning import model_manager as mm_mod
    mm = _new_manager(temp_models_dir)

    # Flip safe_test_mode off so _provision_rugpt3_default executes
    mm.safe_test_mode = False

    # snapshot_download must NOT be called if local model exists
    called = {"count": 0}

    def fake_snapshot_download(**kwargs):
        called["count"] += 1
        raise AssertionError("snapshot_download should not be called when local model exists")

    monkeypatch.setattr(mm_mod, "snapshot_download", fake_snapshot_download, raising=True)

    # Ensure offline flags are not set
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    # Execute
    mm._provision_rugpt3_default()

    # Validate: no download attempted
    assert called["count"] == 0

    # Validate: alias registered and points to our local dir
    meta = mm.model_metadata.get("default_text_gen")
    assert meta is not None
    assert os.path.normpath(meta.model_path) == os.path.normpath(target_dir)


def test_provision_respects_offline(monkeypatch, temp_models_dir):
    # No local model present, but offline -> should skip download
    local_name = "rugpt3_large"
    target_dir = os.path.join(temp_models_dir, local_name)
    os.makedirs(target_dir, exist_ok=True)

    from eva_ai.mlearning import model_manager as mm_mod
    mm = _new_manager(temp_models_dir)
    mm.safe_test_mode = False

    called = {"count": 0}

    def fake_snapshot_download(**kwargs):
        called["count"] += 1
        raise AssertionError("snapshot_download should not be called in offline mode")

    monkeypatch.setattr(mm_mod, "snapshot_download", fake_snapshot_download, raising=True)

    # Set offline env
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    mm._provision_rugpt3_default()

    # Ensure no download attempted
    assert called["count"] == 0

    # Alias still registered to target_dir (even if contents incomplete)
    meta = mm.model_metadata.get("default_text_gen")
    assert meta is not None
    assert os.path.normpath(meta.model_path) == os.path.normpath(target_dir)


def test_provision_downloads_when_missing_and_online(monkeypatch, temp_models_dir):
    # Missing local minimal files and online -> should attempt snapshot_download
    local_name = "rugpt3_large"
    target_dir = os.path.join(temp_models_dir, local_name)
    os.makedirs(target_dir, exist_ok=True)

    from eva_ai.mlearning import model_manager as mm_mod
    mm = _new_manager(temp_models_dir)
    mm.safe_test_mode = False

    call_args = {}

    def fake_snapshot_download(**kwargs):
        # Simulate download by creating minimal files according to allow_patterns
        allow = kwargs.get("allow_patterns") or []
        # Always create the minimal viable set
        _make_minimal_hf_model(target_dir, tokenizer_variant="json")
        call_args.update(kwargs)
        return target_dir

    monkeypatch.setattr(mm_mod, "snapshot_download", fake_snapshot_download, raising=True)

    # Ensure online
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    mm._provision_rugpt3_default()

    # Validate: download attempted
    assert call_args.get("local_dir") == target_dir
    assert os.path.isfile(os.path.join(target_dir, "config.json"))
    assert os.path.isfile(os.path.join(target_dir, "pytorch_model.bin"))
    assert os.path.isfile(os.path.join(target_dir, "tokenizer.json"))

    # Alias registered to target_dir
    meta = mm.model_metadata.get("default_text_gen")
    assert meta is not None
    assert os.path.normpath(meta.model_path) == os.path.normpath(target_dir)
