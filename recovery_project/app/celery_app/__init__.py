from celery import Celery
from celery_app.config import CELERY_CONFIG

celery_app = Celery("simple_rag")
celery_app.config_from_object(CELERY_CONFIG)
celery_app.autodiscover_tasks(["model_app.tasks"])
