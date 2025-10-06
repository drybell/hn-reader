from core.datatypes.sequence import Sequence
from core.datatypes.timestamp import Timestamp

from pydantic import BaseModel, Field, computed_field
from enum import StrEnum
from typing import TypeVar, Any, Literal

HackerT  = TypeVar("T", bound=BaseModel)
StoryId  = TypeVar("StoryId", bound=int)
UserId   = TypeVar("UserId", bound=str)

class ItemType(StrEnum):
    STORY = 'story'
    COMMENT = 'comment'
    JOB = 'job'

class ItemPath(StrEnum):
    ITEM = "item/"
    USER = "user/"

    MAXITEM     = "maxitem.json"
    TOPSTORIES  = "topstories.json"
    NEWSTORIES  = "newstories.json"
    BESTSTORIES = "beststories.json"
    JOBSTORIES  = "jobstories.json"
    ASKSTORIES  = "askstories.json"
    SHOWSTORIES = "showstories.json"

    UPDATES     = "updates.json"

class ResponseError(BaseModel):
    status    : int
    message   : str
    content   : Any | None = None
    error     : Any | None = None
    raw       : Any | None = None

    @computed_field
    @property
    def error_str(self) -> str | None:
        if self.error is not None:
            return str(self.error)

    @computed_field
    @property
    def error_cls(self) -> str | None:
        if self.error is not None:
            return self.error.__class__.__name__

class Item(BaseModel):
    id: StoryId
    type: ItemType | None = None
    deleted: bool | None = None
    by: str | None = None
    time: int | None = None
    text: str | None = None
    dead: bool | None = None
    parent: int | None = None
    poll: int | None = None
    kids: Sequence[int] | None = None
    url: str | None = None
    score: int | None = None
    title: str | None = None
    parts: Sequence[int] | None = None
    descendants: int | None = None

class Story(Item):
    by : str
    descendants : int
    kids : Sequence[int] | None = None
    score : int
    time : int
    title : str
    type : Literal[ItemType.STORY] = ItemType.STORY

class Comment(Item):
    by : str
    kids : Sequence[int] | None = None
    parent : int
    text : str
    time : int
    type : Literal[ItemType.COMMENT] = ItemType.COMMENT

class DeletedComment(Comment):
    by : str
    kids : Sequence[int] | None = None
    parent : int
    text : None = None
    deleted : bool = True
    time : int
    type : Literal[ItemType.COMMENT] = ItemType.COMMENT

class Job(Item):
    by : str
    kids : Sequence[int] | None = None
    score : int
    time : int
    title : str
    url : str | None = None
    type : Literal[ItemType.JOB] = ItemType.JOB

class Ids(BaseModel):
    stories : Sequence[StoryId]

class User(BaseModel):
    id: UserId
    created: int | None = Field(
        default=None, description='Creation date of the user, in Unix Time'
    )
    karma: int | None = Field(default=None, description="The user's karma")
    about: str | None = Field(
        default=None, description="The user's optional self-description. HTML"
    )
    submitted: Sequence[int] | None = Field(
        default=None, description="Sequence of the user's stories, polls and comments"
    )

ItemT = (
    Story
    | DeletedComment
    | Comment
    | Job
    | Item
)

class ItemWrapper(BaseModel):
    item : ItemT = Field(union_mode='left_to_right')

class Updates(BaseModel):
    items    : Sequence[StoryId]
    profiles : Sequence[UserId]

class Updated(BaseModel):
    items    : Sequence[ItemT]
    profiles : Sequence[User]

