import os
import time

import torch

from cogniflex.adapters.torch_adapter import TorchBatchAdapter, Meta
from cogniflex.runtime.worker_pool import InferenceWorkerPool


MODEL_FN = "cogniflex.runtime.simple_model.example_model_fn"


def make_item(seq, idx):
    return {
        "input_ids": torch.tensor(seq, dtype=torch.long),
        "_meta": Meta(idx=idx, length=len(seq)),
    }


def test_torch_pipeline_end_to_end():
    adapter = TorchBatchAdapter(max_items=4, max_tokens=64, timeout_ms=1)
    data = [
        [1, 2, 3],
        [10, 20],
        [7],
        [5, 5, 5, 5],
        [100],
    ]

    for i, seq in enumerate(data):
        adapter.push(make_item(seq, i))

    b1 = adapter.try_pop_batch()
    assert b1 is not None
    b2 = adapter.flush()
    assert b2 is not None

    pool = InferenceWorkerPool(
        model_fn_path=MODEL_FN,
        num_workers=2,
        torch_threads=1,
        interop_threads=1,
    )

    with pool.running():
        ids = [pool.submit(b) for b in (b1, b2)]
        res = []
        for _ in ids:
            bid, out = pool.recv()
            assert not isinstance(out, Exception)
            assert "logits" in out
            res.append(out)

    # Validate shapes and meta indices ordering
    for out in res:
        logits = out["logits"]
        assert logits.dim() == 2 and logits.shape[1] == 1
        meta_idx = out["meta_idx"]
        assert meta_idx.numel() == logits.shape[0]
