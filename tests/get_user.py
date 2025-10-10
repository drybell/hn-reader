import loader
from core.datatypes.sequence import Sequence
from analysis.db import query, Queries
from services.client import HNClient

items = query("select * from items order by id desc limit 10")

s = Sequence(items.by)

users = s.apply(HNClient.get)
