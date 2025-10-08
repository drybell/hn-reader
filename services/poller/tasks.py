from core.datatypes.sequence import Sequence
from core.utils.dt import DT

from config import settings

from services.client import HNClient

from models.db.base import Seeding

from models.base import (
    ItemT, Item, User, ItemWrapper, ResponseError
)

from services.crud import (
    get, post, update_seeding, init_seeding
)

from celery import shared_task
from celery.utils.log import get_task_logger

from functools import wraps

logger = get_task_logger(__name__)

def serialize(Model):
    def decorator(fn):
        @wraps(fn)
        def wrapper(data: dict, *args, **kwargs):
            if Model == ItemWrapper:
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
def fetch(id : int):
    def persist_item():
        persist.delay(
            HNClient.get(id).model_dump(exclude_none=True)
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
        logger.warning(f"items.{item.id} fetch cooldown in effect for {(cd_expiration - now).total_seconds():.2f} more seconds...")
        return

    persist_item()

@shared_task(name="services.poller.tasks.persist")
@serialize(ItemWrapper)
def persist(item : ItemT | ResponseError):
    result = post(item)

    if isinstance(result, ResponseError):
        logger.warning(f"Failed to post item: ResponseError returned -> {result.model_dump(exclude_none=True)}")
        return

    if item.kids:
        logger.info(
            f"adding {item.kids.length()} ids, min={min(item.kids)}, max={max(item.kids)} to fetch"
        )

        item.kids.apply(fetch.delay)

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
