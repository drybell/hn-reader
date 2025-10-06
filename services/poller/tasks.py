from celery import shared_task

from services.client import HNClient
from services.translator import Translator

from models.base import (
    ItemT, Item, User, ItemWrapper
)

from services.crud import (
    post
)

from functools import wraps

def serialize(Model):
    def decorator(fn):
        @wraps(fn)
        def wrapper(data: dict, *args, **kwargs):
            if model == ItemWrapper:
                instance = Model(item=data).item
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

    items.apply(persist.delay)

@shared_task(name="services.poller.tasks.fetch")
def fetch(id : int):
    persist.delay(
        HNClient.get(id).model_dump(exclude_none=True)
    )

@shared_task(name="services.poller.tasks.persist")
@serialize(ItemWrapper)
def persist(item : ItemT):
    post(item)

    if item.kids:
        item.kids.apply(fetch.delay)
