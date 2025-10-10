from core.datatypes.sequence import Sequence
from core.utils.timer import Timer

from services.db import DB

from pydantic import BaseModel, Field
from psycopg import sql

from typing import Any

import pandas as pd
import time

def query(cmd, params=None):
    return pd.read_sql(
        cmd.as_string() if not isinstance(cmd, str) else cmd
        , DB.ENGINE
        , params=params
        , coerce_float=False
    )


class ItemFilter(BaseModel):
    column : str
    op     : str
    value  : Any

    def to_sql(self) -> sql.SQL:
        return sql.SQL("{field} {op} {value}").format(
            field=sql.Identifier(self.column)
            , op=sql.SQL(self.op)
            , value=sql.Literal(self.value)
        )

class IFilter:
    class Eq(ItemFilter):
        op : str = '='

    class Neq(ItemFilter):
        op : str = '!='

    class Ge(ItemFilter):
        op : str = '>='

    class Le(ItemFilter):
        op : str = '<='

    class Gt(ItemFilter):
        op : str = '>'

    class Lt(ItemFilter):
        op : str = '<'

    class Ilike(ItemFilter):
        op : str = 'ilike'

class ItemFilters(BaseModel):
    filters : Sequence[ItemFilter]

    def to_sql(self) -> sql.SQL:
        return sql.SQL(" AND ").join(
            self.filters.apply(lambda f: f.to_sql()).to_list()
        )

NO_COMMENTS = [
    IFilter.Neq(column='type', value='comment')
]

class PresetIFilters:
    NO_COMMENTS = NO_COMMENTS
    NORMAL_SCORES = [
        *NO_COMMENTS
        , IFilter.Ge(column='score', value=75)
    ]
    HIGH_SCORES = [
        *NO_COMMENTS
        , IFilter.Ge(column='score', value=150)
    ]

class TimedeltaComponent(BaseModel):
    d   : int = Field(..., alias="days")
    h   : int = Field(..., alias="hours")
    m   : int = Field(..., alias="minutes")
    s   : int = Field(..., alias="seconds")
    ms  : int = Field(..., alias="milliseconds")
    mcs : int = Field(..., alias="microseconds")
    ns  : int = Field(..., alias="nanoseconds")

    @classmethod
    def from_td(cls, td: pd.Timedelta):
        return cls(**td.components._asdict())

    def condensed(
        self, **kw
    ) -> dict[str, int]:
        return {
            k: v
            for k, v in self.model_dump(**kw).items()
            if v != 0
        }

class WriteRateStats(BaseModel):
    completion_duration : TimedeltaComponent
    elapsed_time        : TimedeltaComponent
    delta_results       : int

class Queries:
    class Utils:
        class Timing:
            @staticmethod
            def write_rate(table : str, delay : int = 60) -> WriteRateStats:
                maxid = query(
                    f"select max(id) as maxid from {table}"
                ).maxid.iloc[0]

                t1, et = Timer.timeit(
                    query
                    , f"select count(id) as total from {table}"
                )

                time.sleep(delay)

                t2, et2 = Timer.timeit(
                    query
                    , f"select count(id) as total from {table}"
                )

                diff = (t2.total.iloc[0] - t1.total.iloc[0])
                dt = (et2.end - et.end)

                return WriteRateStats(
                    completion_duration=TimedeltaComponent.from_td(
                        pd.Timedelta(
                            seconds=maxid / (diff / dt.total_seconds())
                        )
                    )
                    , elapsed_time=TimedeltaComponent.from_td(dt)
                    , delta_results=diff
                )

        class IDs:
            @staticmethod
            def find_missing(
                table
                , min_id=None
                , max_id=None
            ):
                df = query(f"select id from {table} order by id")
                existing = df.id.dropna().astype(int)

                min_id = min_id if min_id else existing.min()
                max_id = max_id if max_id else existing.max()

                expected = pd.Series(range(min_id, max_id + 1))
                return (
                    expected[
                        ~expected.isin(existing)
                    ]
                    , df
                )

    class Sentiment:
        class Categories:
            @staticmethod
            def get_posts(
                filters : Sequence[ItemFilter]
                , limit : int = 1000
            ) -> pd.DataFrame:
                stmt = sql.SQL("""
                SELECT
                    id, title
                FROM items
                WHERE {filters}
                ORDER BY id desc
                LIMIT {limit}
                """).format(
                    filters=ItemFilters(filters=filters).to_sql()
                    , limit=sql.Literal(limit)
                )

                return query(stmt)
