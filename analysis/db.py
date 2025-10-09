from core.datatypes.sequence import Sequence

from services.db import DB

from pydantic import BaseModel
from psycopg import sql

from typing import Any

import pandas as pd

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

class Queries:
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
