"""Celery entrypoint that installs multilingual routing before task registration."""

import multilingual_server  # noqa: F401  # installs patches on the shared server module
from celery_worker import celery_app

__all__ = ["celery_app"]
