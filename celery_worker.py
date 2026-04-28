import os

from celery import Celery

from server import CELERY_BROKER_URL, CELERY_QUEUE, CELERY_RESULT_BACKEND, logger, synthesize_payload

celery_app = Celery(
    "chatterbox_turbo",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_default_queue=CELERY_QUEUE,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_track_started=True,
    broker_connection_retry_on_startup=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)


@celery_app.task(name="chatterbox_turbo.synthesize")
def synthesize(payload: dict[str, object]) -> dict[str, object]:
    logger.info(
        "Celery synthesis task received (queue=%s, hostname=%s).",
        CELERY_QUEUE,
        os.uname().nodename,
    )
    return synthesize_payload(payload)
