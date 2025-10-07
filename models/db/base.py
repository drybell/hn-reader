from core.datatypes.timestamp import Timestamp
from core.utils.dt import DT

from sqlmodel import (
    Column, Field, SQLModel, Relationship, Integer
)

from sqlalchemy.dialects.postgresql import ARRAY


class Item(SQLModel, table=True):
    __tablename__ = 'items'

    id: int = Field(primary_key=True)
    type: str = Field(index=True)
    deleted: bool | None = Field(default=None)
    by: str | None = Field(default=None)
    time : int | None = Field(default=None)
    text : str | None = Field(default=None)
    parent : int | None = Field(default=None)
    poll : int | None = Field(default=None)
    kids : list[int] | None = Field(
        default=None
        , sa_column=Column(ARRAY(Integer))
    )
    url : str | None = Field(default=None)
    score : int | None = Field(default=None)
    title : str | None = Field(default=None)
    parts : list[int] | None = Field(
        default=None
        , sa_column=Column(ARRAY(Integer))
    )
    descendants : int | None = Field(default=None)

    created_ts: Timestamp = Field(default_factory=DT.now)

class User(SQLModel, table=True):
    id: str = Field(primary_key=True)
    created: int | None = Field(default=None)
    karma: int | None = Field(default=None)
    about: str | None = Field(default=None)
    submitted: list[int] | None = Field(
        default=None
        , sa_column=Column(ARRAY(Integer))
    )

class Seeding(SQLModel, table=True):
    __tablename__ = 'seeding_queue'

    id : int | None = Field(default=None, primary_key=True)
    currid : int
    batch_size : int
    start_date : Timestamp
