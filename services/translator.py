from models.db.base import (
    User as UserDB
    , Item as ItemDB
)

from models.base import (
    ItemT, User
)

from typing import Any

class Translator:

    @staticmethod
    def _translate(obj, Model):
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
        def item(obj : ItemDB) -> Item:
            return Translator._translate(
                obj, Item
            )

        @staticmethod
        def user(obj : UserDB) -> User:
            return Translator._translate(
                obj, User
            )
