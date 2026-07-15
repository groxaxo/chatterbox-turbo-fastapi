from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_worker_launcher_defaults_to_resident_warm_model():
    script = (ROOT / "run_celery_worker.sh").read_text()

    assert 'WORKER_LAZY_LOAD_MODEL="${WORKER_LAZY_LOAD_MODEL:-0}"' in script
    assert 'WORKER_MODEL_IDLE_UNLOAD_SECONDS="${WORKER_MODEL_IDLE_UNLOAD_SECONDS:-${MODEL_IDLE_UNLOAD_SECONDS:-0}}"' in script
    assert 'WORKER_STARTUP_WARMUP="${WORKER_STARTUP_WARMUP:-1}"' in script
    assert "export CHATTERBOX_WORKER_BOOTSTRAP=1" in script


def test_installer_persists_resident_worker_policy():
    installer = (ROOT / "install_systemd_services.sh").read_text()

    assert "WORKER_LAZY_LOAD_MODEL=0" in installer
    assert "WORKER_MODEL_IDLE_UNLOAD_SECONDS=0" in installer
    assert "WORKER_STARTUP_WARMUP=1" in installer


def test_application_does_not_add_redundant_cuda_synchronization():
    server = (ROOT / "server.py").read_text()
    multilingual = (ROOT / "multilingual_runtime.py").read_text()

    assert "torch.cuda.synchronize()" not in server
    assert "torch.cuda.synchronize()" not in multilingual
