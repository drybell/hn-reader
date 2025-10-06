from services.db import DB
from models.db.base import Item, User

from services.client import HNClient
from services.translator import Translator

DB.create_all()
s = DB.new_session()

def create_test_item(session=None):
    top = HNClient.best()
    item = Item(**top[0].model_dump())
    session.add(item)
    session.commit()
    session.refresh(item)
    return item

def select_test_item():
    return s
