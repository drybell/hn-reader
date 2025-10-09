from core.datatypes.sequence import Sequence

from pydantic import BaseModel

from enum import StrEnum

class ConfidenceLevel(StrEnum):
    HIGH   = 'high'
    MEDIUM = 'medium'
    LOW    = 'low'

class Category(BaseModel):
    label       : str
    description : str
    confidence  : float
    level       : ConfidenceLevel

class TaggedPost(BaseModel):
    id    : int
    title : str
    tags  : Sequence[Category]

    def condensed_fmt(self) -> dict:
        return {
            'id': self.id
            , 'title': self.title[:50] + '...' if len(self.title) > 50 else self.title
            , 'labels': self.tags.label.to_list()
            , 'confidence': self.tags.confidence.to_list()
            , 'levels': self.tags.level.value.to_list()
        }
