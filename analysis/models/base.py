from core.datatypes.sequence import Sequence

from models.base import ItemT

from pydantic import BaseModel

class Post(BaseModel):
    id     : int
    title  : str
    text   : str | None = None
    score  : int
    url    : str | None = None
    author : str

class PostCollection(BaseModel):
    posts : Sequence[Post]

    @classmethod
    def from_items(cls, items : Sequence[ItemT]):
        return cls(
            posts=items.model_dump().to_list()
        )

    def to_title_df(self):
        return self.posts.model_dump().to_frame()
