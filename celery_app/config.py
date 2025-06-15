import os

CELERY_CONFIG = {
    "imports": ("model_app.tasks",),  # Ensure tasks are discovered
    "worker_send_task_events": True,  # Enable task events
    "task_send_sent_event": True,  # Enable sent events
    "broker_url": os.getenv("CELERY_BROKER_URL", "amqp://guest:guest@rabbitmq:5672//"),
    "result_backend": os.getenv("CELERY_RESULT_BACKEND", "rpc://"),
    "task_serializer": "json",
    "result_serializer": "json",
    "accept_content": ["json"],
    "timezone": "UTC",
    "enable_utc": True,
    "task_default_queue": "default",
    "task_routes": {
        "model_app.tasks.text_to_embeddings": {
            "queue": "embeddings"
        }
    },
    "beat_schedule": {
        # Add any periodic tasks here if needed
    }
}
