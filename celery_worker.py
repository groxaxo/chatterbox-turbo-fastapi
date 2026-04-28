import os

from celery import Celery

from server import (
    CELERY_BROKER_URL,
    CELERY_QUEUE,
    CELERY_RESULT_BACKEND,
    logger,
    synthesize_payload,
    synthesize_single_chunk_payload,
)

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


@celery_app.task(name="chatterbox_turbo.synthesize_chunk")
def synthesize_chunk(payload: dict[str, object]) -> dict[str, object]:
    """
    Synthesize a single pre-split text chunk and return WAV as base64.
    Called in parallel by the API server when a request contains multiple sentences.
    """
    logger.info(
        "Celery chunk task received (queue=%s, hostname=%s, text_len=%d).",
        CELERY_QUEUE,
        os.uname().nodename,
        len(str(payload.get("text", ""))),
    )
    return synthesize_single_chunk_payload(payload)
