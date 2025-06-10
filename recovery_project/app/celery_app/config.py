import os

CELERY_CONFIG = {
    "broker_url": os.getenv("CELERY_BROKER_URL", "amqp://guest:guest@rabbitmq:5672//"),
    "result_backend": os.getenv("CELERY_RESULT_BACKEND", "rpc://"),
    "task_serializer": "json",
    "result_serializer": "json",
    "accept_content": ["json"],
    "timezone": "UTC",
    "enable_utc": True,
    "task_default_queue": "default",
    "task_routes": {
        os.getenv("CELERY_EMBEDDINGS_TASK_NAME", "model_app.tasks.text_to_embeddings"): {
            "queue": os.getenv("CELERY_EMBEDDINGS_QUEUE", "embeddings")
        }
    },
    "beat_schedule": {
        # Add any periodic tasks here if needed
    }
}
