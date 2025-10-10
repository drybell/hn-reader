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
    , "services.poller.tasks.seeding": {
        'queue': 'seeding'
    }
    , "services.poller.tasks.retry": {
        'queue': 'retries'
    }
}

@app.on_after_configure.connect
def setup_periodic_tasks(sender: Celery, **kwargs):
    sender.add_periodic_task(
        120.0
        , services.poller.tasks.refresh.s('best')
        , name="refresh TopHN posts"
    )

    sender.add_periodic_task(
        125.0
        , services.poller.tasks.refresh.s('jobs')
        , name="refresh Job postings"
    )

    sender.add_periodic_task(
        130.0
        , services.poller.tasks.refresh.s('showhn')
        , name="refresh ShowHN posts"
    )

    sender.add_periodic_task(
        200.0
        , services.poller.tasks.refresh.s('new')
        , name="refresh new HN posts"
    )

    sender.add_periodic_task(
        150.0
        , services.poller.tasks.seeding.s()
        , name="seed all ids backwards"
    )

#app.autodiscover_tasks()
