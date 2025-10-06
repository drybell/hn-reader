from models.db.base import (
    User as UserDB
    , Item as ItemDB
)

from models.base import (
    ItemWrapper, ItemT, User
)

from typing import Any

class Translator:

    @staticmethod
    def _translate(obj, Model) -> Any | None:
        if obj is None:
            return None

        if Model == ItemWrapper:
            return Model(item=obj.model_dump())
        else:
            return Model(**obj.model_dump())

    class DB:
        @staticmethod
        def item(obj : ItemT) -> ItemDB:
            return Translator._translate(
                obj, ItemDB
            )

        @staticmethod
        def user(obj : User) -> UserDB:
            return Translator._translate(
                obj, UserDB
            )

    class Generic:
        @staticmethod
        def item(obj : ItemDB) -> ItemT | None:
            item = Translator._translate(
                obj, ItemWrapper
            )

            if item is None:
                return item

            return item.item

        @staticmethod
        def user(obj : UserDB) -> User:
            return Translator._translate(
                obj, User
            )
