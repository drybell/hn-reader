from core.datatypes.sequence import Sequence
from core.datatypes.timestamp import Timestamp
from core.utils.dt import DT

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
    POLL = 'poll'
    POLL_OPTION = 'pollopt'

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
    args      : Any
    status    : int
    message   : str
    content   : Any | None = None
    error     : Any | None = None
    raw       : Any | None = None
    retries   : int | None = None

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

    def condensed(self) -> dict:
        return {
            'args': self.args
            , 'status': self.status
            , 'message': self.message
            , 'error': str(self.error)
        }

    def model_dump(self, *args, **kw) -> dict:
        base = super().model_dump(*args, **kw)
        base.pop('error', None)
        return {
            **base
            , 'error_str': self.error_str
            , 'error_cls': self.error_cls
        }

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
    last_fetch_ts : Timestamp = Field(default_factory=DT.now)
    retry_count : int | None = None

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
    by : str | None = None
    text : None = None
    deleted : bool = True
    type : Literal[ItemType.COMMENT] = ItemType.COMMENT

class FlaggedComment(Comment):
    by : str | None = None
    text : Literal['[flagged]'] = '[flagged]'
    type : Literal[ItemType.COMMENT] = ItemType.COMMENT

class DeadComment(Comment):
    by : str | None = None
    text : Literal['[dead]'] = '[dead]'
    type : Literal[ItemType.COMMENT] = ItemType.COMMENT

class Job(Item):
    by : str
    score : int
    time : int
    title : str
    url : str | None = None
    type : Literal[ItemType.JOB] = ItemType.JOB

class Poll(Item):
    by : str
    parts : Sequence[int]
    score : int
    time : int
    title : str
    type : Literal[ItemType.POLL] = ItemType.POLL

class PollOption(Item):
    by : str
    poll : int
    score : int
    time : int
    text : str
    type : Literal[ItemType.POLL_OPTION] = ItemType.POLL_OPTION

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
    | FlaggedComment
    | DeadComment
    | DeletedComment
    | Comment
    | Job
    | Poll
    | PollOption
    | Item
)

CommentT = (
    FlaggedComment
    | DeadComment
    | DeletedComment
    | Comment
)

HackerItemT = (
    ItemT
    | User
    | ResponseError
)

class ItemWrapper(BaseModel):
    item : ItemT = Field(union_mode='left_to_right')

class HNConsumer(BaseModel):
    item : HackerItemT = Field(union_mode='left_to_right')

    @classmethod
    def consume(cls, item : dict | HackerItemT):
        match item:
            case dict():
                return cls(item=item).item
            case _:
                return cls(item=item.model_dump()).item

class Updates(BaseModel):
    items    : Sequence[StoryId]
    profiles : Sequence[UserId]

class Updated(BaseModel):
    items    : Sequence[ItemT]
    profiles : Sequence[User]

class CommentThread(Comment):
    """Single comment in the thread tree."""
    depth: int
    path: Sequence[int]

class DeletedCommentThread(DeletedComment):
    """Single deleted comment in the thread tree."""
    depth: int
    path: Sequence[int]

class FlaggedCommentThread(FlaggedComment):
    """Single deleted comment in the thread tree."""
    depth: int
    path: Sequence[int]

class DeadCommentThread(DeadComment):
    """Single deleted comment in the thread tree."""
    depth: int
    path: Sequence[int]

class StoryRoot(Story):
    """Top-level story in the thread tree."""
    depth: int
    path: Sequence[int]

class JobRoot(Job):
    """Top-level job in the thread tree."""
    depth: int
    path: Sequence[int]

ThreadT = (
    StoryRoot
    | JobRoot
    | DeletedCommentThread
    | DeadCommentThread
    | FlaggedCommentThread
    | CommentThread
)

CommentThreadT = (
    DeletedCommentThread
    | DeadCommentThread
    | FlaggedCommentThread
    | CommentThread
)

class ThreadPage(BaseModel):
    """Paginated thread response."""
    id: str = Field(
        description="Unique hash of story_id, page, and page_size"
    )
    items: Sequence[ThreadT]
    total: int
    page: int
    page_size: int
    has_next: bool

class CommentThreadExpanded(BaseModel):
    """Response for expanding a specific comment's children."""
    id: str = Field(
        description="Unique hash of parent_id, page, and page_size"
    )
    parent: CommentT = Field(union_mode='left_to_right')
    children: Sequence[CommentThreadT] = Field(union_mode='left_to_right')
    total_children: int
    page: int
    page_size: int
    has_next: bool

    def comments(self) -> Sequence[CommentThreadT | CommentT]:
        return Sequence([
            self.parent, *self.children
        ])
