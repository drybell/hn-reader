from config import settings

import services.poller.tasks

from celery import Celery
from celery.schedules import crontab

app = Celery(
    "HN Poller",
    **settings.redis.celery_config.model_dump()
)

app.conf.task_routes = {
    "services.poller.tasks.fetch": {
        'queue': 'ids_to_fetch'
    }
    , "services.poller.tasks.persist": {
        'queue': 'fetched_items'
    }
    , "services.poller.tasks.refresh": {
        'queue': 'refreshes'
    }
}

@app.on_after_configure.connect
def setup_periodic_tasks(sender: Celery, **kwargs):
    sender.add_periodic_task(
        120.0
        , services.poller.tasks.refresh.s('best')
        , name="refresh TopHN posts"
    )

#app.autodiscover_tasks()
