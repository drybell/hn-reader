from services.db import DB
from models.db.base import Item, User

from services.client import HNClient
from services.translator import Translator

import services.crud as crud

from sqlmodel import select

DB.create_all()
s = DB.new_session()

def create_test_item(session=None):
    top = HNClient.best()
    item = Item(**top[0].model_dump())
    s.add(item)
    s.commit()
    s.refresh(item)
    return item

def select_test_item():
    return Translator.Generic.item(
        s.exec(
            select(Item).where(Item.id == 45483386)
        ).first()
    )

def get_item(session, id):
    return crud.get_item_by_id(
        session=session
        , id=id
        , translate=True
    )

def post_item(session, item):
    return crud.create_item(
        session=session, item=item
    )

def update_item(session, item):
    return crud.update_item(
        session=session
        , itemdb=get_item(session, item.id)
        , update_item=item
    )

def init_seeding():
    return crud.init_seeding()
