import pytest

pytest.importorskip("celery")

import celery_worker


def test_worker_bootstrap_propagates_model_load_failure(monkeypatch):
    def fail():
        raise RuntimeError("model load failed")

    monkeypatch.setattr(celery_worker, "warm_worker_model_if_needed", fail)

    with pytest.raises(RuntimeError, match="model load failed"):
        celery_worker.bootstrap_worker_runtime()


def test_worker_runtime_probe_uses_process_status(monkeypatch):
    expected = {"model_loaded": True, "process_warmup_complete": True}
    monkeypatch.setattr(celery_worker, "runtime_status", lambda include_sensitive: expected)

    assert celery_worker.worker_runtime_status.run() == expected
