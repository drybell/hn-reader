from core.datatypes.sequence import Sequence
from core.datatypes.timestamp import Timestamp

from pydantic import BaseModel

from enum import StrEnum, IntEnum

class DateRange(BaseModel):
    start : Timestamp | None = None
    end   : Timestamp | None = None

class Greed(IntEnum):
    LAZY             = 35
    RESERVED         = 50
    NORMAL           = 75
    AGGRESSIVE       = 89
    MAX_CAPABILITIES = 100

class SeedLevel(IntEnum):
    SPECIFIC_ITEM = 0x0000F
    BASE_ITEMS    = 0x000F0
    ROOT_COMMENTS = 0x00F00
    LONG_THREADS  = 0x0F000
    ALL_THREADS   = 0xF0000
    EVERYTHING    = 0xFFFF0

class DeletedFilter(IntEnum):
    COMMENTS = 0b0001
    STORIES  = 0b0010
    JOBS     = 0b0100
    USERS    = 0b1000
    ALL      = 0b1111

class ExclusionFilter(IntEnum):
    COMMENTS = 0b0001
    STORIES  = 0b0010
    JOBS     = 0b0100
    USERS    = 0b1000
    ALL      = 0b1111

class SeedDirection(IntEnum):
    BACKWARD = -1
    FORWARD  = 1

class ConfigEncoder:
    @staticmethod
    def collapse(values):
        match values:
            case Sequence():
                return values.reduce(lambda x, y: x | y)
            case int():
                return values

    @staticmethod
    def seeding(*args):
        (
            level, greed, direction
            , skip_deleted, exclude
        ) = Sequence(args).apply(
            ConfigEncoder.collapse
        )

        value = 0
        value |= (
            level & 0xFFFFF
        ) << 16
        value |= (
            greed & 0x7F
        ) << 9
        value |= (
            1
            if direction == SeedDirection.FORWARD
            else 0
        ) << 8
        value |= (
            skip_deleted & 0xF
        ) << 4
        value |= (
            exclude & 0xF
        )
        return value


class Presets:
    class SeedConfig(IntEnum):
        DO_NOTHING = 0
        AGGRESSIVE_JOB_HUNT = ConfigEncoder.seeding(
            SeedLevel.ALL_THREADS
            , Greed.AGGRESSIVE
            , SeedDirection.FORWARD
            , Sequence([
                DeletedFilter.COMMENTS
                , DeletedFilter.JOBS
            ])
            , Sequence([
                ExclusionFilter.USERS
                , ExclusionFilter.STORIES
            ])
        )

class SeedFlag(BaseModel):
    value : Presets.SeedConfig | int

class SeedConfig(BaseModel):
    level        : Sequence[SeedLevel] | SeedLevel
    skip_deleted : Sequence[DeletedFilter] | DeletedFilter
    exclude      : Sequence[ExclusionFilter] | ExclusionFilter
    direction    : Sequence[SeedDirection] | SeedDirection
    greed        : Greed

    def encode(self) -> SeedFlag:
        """
        Encodes the enum fields of SeedConfig into a single 36-bit integer.
        Layout (left to right):
        [20 bits: level][7 bits: greed][1 bit: direction][4 bits: skip_deleted][4 bits: exclude]
        """
        return ConfigEncoder.seeding(
            self.level
            , self.greed
            , self.direction
            , self.skip_deleted
            , self.exclude
        )

class TaskBehavior(BaseModel):
    dt_range : DateRange
    seeding  : SeedConfig
    updates  : None # TODO
