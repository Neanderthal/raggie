from celery import Celery
from celery_app.config import CELERY_CONFIG
import logging

logger = logging.getLogger(__name__)

celery_app = Celery("simple_rag")
celery_app.config_from_object(CELERY_CONFIG)

# Let autodiscover handle task registration without explicit imports
celery_app.autodiscover_tasks(["model_app.tasks"])

logger.info("Celery app initialized")
