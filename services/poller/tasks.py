from core.datatypes.sequence import Sequence
from core.utils.dt import DT

from config import settings

from services.client import HNClient

from models.db.base import Seeding, User as UserDB

from models.base import (
    ItemT
    , Item
    , User
    , ItemWrapper
    , ResponseError
    , HNConsumer
    , Poll
    , HackerItemT
)

from services.crud import (
    get, post, update_seeding, init_seeding
)

from celery import shared_task
from celery.utils.log import get_task_logger

from functools import wraps

logger = get_task_logger(__name__)

class DelayConfig:
    base_delay  = 60
    max_retries = 5

    @classmethod
    def exponential_backoff(cls, retry_count : int) -> int:
        return cls.base_delay * (2 ** retry_count)

def serialize(Model):
    def decorator(fn):
        @wraps(fn)
        def wrapper(data: dict, *args, **kwargs):
            if Model == HNConsumer:
                instance = HNConsumer.consume(data)
            elif Model == ItemWrapper:
                try:
                    instance = Model(item=data).item
                except Exception as e:
                    # TODO: better exception handling
                    instance = ResponseError(**data)
            else:
                instance = Model(**data)
            return fn(instance, *args, **kwargs)
        return wrapper
    return decorator

@shared_task(name="services.poller.tasks.refresh")
def refresh(rtype : str):
    match rtype:
        case 'new':
            items = HNClient.new()
        case 'best':
            items = HNClient.best()
        case 'showhn':
            items = HNClient.showhn()
        case 'askhn':
            items = HNClient.askhn()
        case 'jobs':
            items = HNClient.jobs()

    logger.info(f"refreshing {rtype}")
    items.stories.apply(fetch.delay)

@shared_task(name="services.poller.tasks.fetch")
def fetch(id : int | str, retry_count : int = 0):
    def persist_item():
        persist.delay(
            HNClient.get(
                id, retry_count=retry_count
            ).model_dump(exclude_none=True)
        )

    item = get(id)

    if item is None:
        persist_item()
        return

    cd_expiration = DT.add_delta(
        item.last_fetch_ts
        , settings.client.item_fetch_cooldown
    )

    now = DT.now(tz=None)

    if cd_expiration >= now:
        logger.warning(f"{item.__class__.__name__}.{item.id} fetch cooldown in effect for {(cd_expiration - now).total_seconds():.2f} more seconds...")
        return

    persist_item()

@shared_task(name="services.poller.tasks.retry")
def retry(id : int, retry_count : int = 0):
    if retry_count is None:
        retry_count = 0

    if retry_count >= DelayConfig.max_retries:
        logger.error(
            f"items.{id} failed after {DelayConfig.max_retries}... Giving up."
        )
        return

    fetch.apply_async(
        args=[id]
        , kwargs={
            'retry_count': retry_count + 1
        }
        , queue='ids_to_fetch'
        , countdown=DelayConfig.exponential_backoff(
            retry_count
        )
    )

@shared_task(name="services.poller.tasks.persist")
@serialize(HNConsumer)
def persist(item : HackerItemT):
    def do_fetch(id):
        fetch.apply_async(
            args=[id]
            , queue='ids_to_fetch'
        )

    def fetch_wrapper(obj):
        match obj:
            case Sequence():
                obj.apply(do_fetch)
            case _:
                do_fetch(obj)

    match item:
        case ResponseError():
            logger.warning(f"[ATTEMPT {item.retries or 0}]: Failed to post item: ResponseError returned -> {item.condensed()}")
            recovered_id = item.args[0]
            retry.delay(recovered_id, retry_count=item.retries)
        case User():
            post(item)

            if item.submitted:
                fetch_wrapper(item.submitted)
        case _:
            post(item)

            fetch_wrapper(item.by)

            if item.kids:
                fetch_wrapper(item.kids)

            if item.parts:
                fetch_wrapper(item.parts)

@shared_task(name="services.poller.tasks.seeding")
def seeding():
    config = init_seeding()

    if config.currid <= 1:
        logger.warning(f"seeding.{config.currid} completed, skipping execution...")
        return

    newid = max(1, config.currid - config.batch_size)

    Sequence(
        range(
            config.currid
            , newid
            , -1
        )
    ).apply(fetch.delay)

    logger.info(f"adding ids [{config.currid}...{newid} to fetch")

    update_seeding(currid=newid, last_execution_ts=DT.now())
