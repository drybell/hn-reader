from core.datatypes.sequence import Sequence

from models.db.base import (
    Item as ItemDB
    , User as UserDB
)

from models.base import (
    Item, ItemT, Story, Comment, Job, User
)

from services.translator import Translator
from services.db import DB

from sqlmodel import Session, select
from sqlalchemy.exc import IntegrityError

def session_lifecycle(
    session : Session
    , item  : ItemDB | UserDB
) -> ItemDB | UserDB:
    session.add(item)
    session.commit()
    session.refresh(item)
    return item

def get_item_by_id(
    *
    , session : Session
    , id : Sequence[int] | int
    , translate : bool = False
) -> ItemDB | ItemT | Sequence[ItemDB] | Sequence[ItemT]:
    match id:
        case Sequence():
            items = session.exec(
                select(ItemDB).where(
                    ItemDB.id.in_(id.to_list())
                )
            )

            if translate:
                return Sequence(items).apply(
                    Translator.Generic.item
                )

            return Sequence(items)
        case int():
            item = session.exec(
                select(ItemDB).where(ItemDB.id == id)
            ).first()

            if translate:
                return Translator.Generic.item(item)

            return item
        case _:
            raise TypeError(f"id: {id} : {type(id)} is not a Sequence or int!")

def get_user_by_id(
    *
    , session : Session
    , id : str
    , translate : bool = False
) -> UserDB | User:
    user = session.exec(
        select(UserDB).where(UserDB.id == id)
    ).first()

    if translate:
        return Translator.Generic.user(user)

    return user

def create_item(*, session: Session, item : ItemT) -> ItemDB:
    dbitem = Translator.DB.item(item)

    try:
        return session_lifecycle(
            session, dbitem
        )
    except IntegrityError:
        session.rollback()
        existing = get_item_by_id(
            session=session, id=item.id
        )

        return update_item(
            session=session
            , itemdb=existing
            , update_item=item
        )

def create_user(*, session: Session, user : User) -> UserDB:
    dbuser = Translator.DB.user(user)

    try:
        return session_lifecycle(
            session, dbuser
        )
    except IntegrityError:
        session.rollback()
        existing = get_user_by_id(
            session=session, id=user.id
        )

        return update_user(
            session=session
            , userdb=existing
            , update_user=item
        )

def update_item(
    *
    , session     : Session
    , itemdb      : ItemDB
    , update_item : ItemT
) -> ItemDB:
    itemdb.sqlmodel_update(
        Translator.DB.item(update_item).model_dump(
            exclude_unset=True
        )
    )

    return session_lifecycle(session, itemdb)

def update_user(
    *
    , session     : Session
    , userdb      : UserDB
    , update_user : User
) -> UserDB:
    userdb.sqlmodel_update(
        Translator.DB.user(update_user).model_dump(
            exclude_unset=True
        )
    )

    return session_lifecycle(session, userdb)

def post(
    data : ItemT | User
) -> ItemDB | UserDB:
    with Session(DB.ENGINE) as session:
        match data:
            case User():
                return create_user(session=session, user=data)
            case _:
                return create_item(session=session, item=data)

def get(
    id : int | str
    , translate : bool = False
) -> ItemDB | ItemT | UserDB | User:
    with Session(DB.ENGINE) as session:
        match id:
            case int():
                return get_item_by_id(
                    session=session
                    , id=id
                    , translate=translate
                )
            case str():
                return get_user_by_id(
                    session=session
                    , id=id
                    , translate=translate
                )
