from core.datatypes.sequence import Sequence
from core.datatypes.timestamp import Timestamp

from config import settings

from models.db.base import (
    Item as ItemDB
    , User as UserDB
    , Seeding
)

from models.base import (
    Item
    , ItemT
    , Story
    , Comment
    , Job
    , User
    , ResponseError
    , CommentThread
    , ThreadPage
    , CommentThreadExpanded
)

from services.translator import Translator
from services.db import DB

from sqlmodel import Session, select, text

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError

import hashlib

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
            , update_user=user
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
    data : ItemT | User | ResponseError
) -> ItemDB | UserDB | ResponseError:
    with Session(DB.ENGINE) as session:
        match data:
            case ResponseError():
                return data
            case User():
                return create_user(session=session, user=data)
            case _:
                return create_item(
                    session=session, item=data
                )

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

# TODO: throw into separate module
class Hashers:
    """
    Generates unique hashes for higher-level models
    based on some parameters used by the caller
    """
    class Utils:
        @staticmethod
        def hash(content : str) -> str:
            return hashlib.md5(content.encode()).hexdigest()

    class ThreadPage:
        @staticmethod
        def id(
            id: int
            , page: int
            , page_size: int
        ) -> str:
            return Hashers.Utils.hash(f"{id}:{page}:{page_size}")

    class CommentThreadExpanded:
        @staticmethod
        def id(
            parent_id: int
            , page: int
            , page_size: int
        ):
            return Hashers.Utils.hash(
                f"{parent_id}:{page}:{page_size}"
            )

def get_story_thread(
    *
    , session: Session
    , id: int
    , page: int = 1
    , page_size: int = 50
    , max_children_per_parent: int | None = None
) -> ThreadPage:
    """
    Fetch paginated comment thread for a story.

    Args:
        session: SQLModel database session
        id: HN story ID
        page: Page number (1-indexed)
        page_size: Number of comments per page
        max_children_per_parent: Limit children returned per parent

    Returns:
        ThreadPage with comments and pagination metadata
    """
    offset = (page - 1) * page_size

    # Build the recursive CTE with optional child limiting
    child_filter = ""

    if max_children_per_parent:
        child_filter = f"""
            AND child_rank <= :max_children
        """

    # f-string ok due to setting it directly (no user-input)
    query = text(f"""
        WITH RECURSIVE comment_tree AS (
          -- Base case: get the story
          SELECT
            i.*
            , 0 as depth
            , ARRAY[i.id] as path
            , 0::bigint as child_rank
          FROM items i
          WHERE i.id = :id

          UNION ALL

          -- Recursive case: get ranked children
          SELECT
            i.*
            , ct.depth + 1
            , ct.path || i.id
            , ROW_NUMBER() OVER (
                PARTITION BY i.parent
                ORDER BY i.time ASC
            ) as child_rank
          FROM items i
          INNER JOIN comment_tree ct ON i.parent = ct.id
          WHERE i.type = 'comment'
            {child_filter}
        )
        , counted AS (
          SELECT COUNT(*) as total FROM comment_tree
        )
        SELECT
          ct.*, c.total
        FROM comment_tree ct
        CROSS JOIN counted c
        ORDER BY ct.path
        LIMIT :limit OFFSET :offset
    """)

    params = {
        "id": id
        , "limit": page_size
        , "offset": offset
    }

    if max_children_per_parent:
        params["max_children"] = max_children_per_parent

    result = session.exec(query, params=params)
    rows = result.fetchall()

    if not rows:
        return None

    total = rows[0].total if rows else 0

    has_next = offset + page_size < total

    return ThreadPage(
        id=Hashers.ThreadPage.id(id, page, page_size)
        , items=[
            row._mapping for row in rows
        ]
        , total=total
        , page=page
        , page_size=page_size
        , has_next=has_next
    )

def get_comment_children(
    *
    , session: Session
    , parent_id: int
    , page: int = 1
    , page_size: int = 50
    , max_depth: int | None = None
) -> CommentThreadExpanded:
    """
    Fetch paginated children of a specific comment.

    Args:
        session: SQLModel database session
        parent_id: Parent comment ID to expand
        page: Page number (1-indexed)
        page_size: Number of comments per page
        max_depth: Optional depth limit for recursion

    Returns:
        CommentThreadExpanded with children and pagination metadata
    """
    offset = (page - 1) * page_size

    depth_filter = ""
    if max_depth is not None:
        depth_filter = "WHERE depth <= :max_depth"

    query = text(f"""
        WITH RECURSIVE comment_tree AS (
          -- Base case: direct children of parent
          SELECT
            i.*
            , 1 as depth
            , ARRAY[i.id] as path
          FROM items i
          WHERE i.parent = :parent_id
            AND i.type = 'comment'

          UNION ALL

          -- Recursive case: descendants
          SELECT
            i.*
            , ct.depth + 1
            , ct.path || i.id
          FROM items i
          INNER JOIN comment_tree ct ON i.parent = ct.id
          WHERE i.type = 'comment'
        )
        , filtered AS (
          SELECT * FROM comment_tree
          {depth_filter}
        )
        , counted AS (
          SELECT COUNT(*) as total FROM filtered
        )
        SELECT
          f.*, c.total
        FROM filtered f
        CROSS JOIN counted c
        ORDER BY f.path
        LIMIT :limit OFFSET :offset
    """)

    params = {
        "parent_id": parent_id
        , "limit": page_size
        , "offset": offset
    }

    # TODO: re-use from get_thread
    if max_depth is not None:
        params["max_depth"] = max_depth

    result = session.exec(query, params=params)
    rows = result.fetchall()

    if not rows:
        return None

    total = rows[0].total if rows else 0

    has_next = offset + page_size < total

    return CommentThreadExpanded(
        id=Hashers.CommentThreadExpanded.id(
            parent_id, page, page_size
        )
        , parent=get_item_by_id(
            session=session
            , id=parent_id
            , translate=True
        )
        , children=[
            row._mapping for row in rows
        ]
        , total_children=total
        , page=page
        , page_size=page_size
        , has_next=has_next
    )

def get_thread(
    id : int
    , page: int = 1
    , page_size: int = 50
    , max_children_per_parent: int | None = None
):
    with Session(DB.ENGINE) as session:
        return get_story_thread(
            session=session
            , id=id
            , page=page
            , page_size=page_size
            , max_children_per_parent=max_children_per_parent
        )

def expand_comment(
    parent_id : int
    , page: int = 1
    , page_size: int = 50
    , max_depth: int | None = None
):
    with Session(DB.ENGINE) as session:
        return get_comment_children(
            session=session
            , parent_id=parent_id
            , page=page
            , page_size=page_size
            , max_depth=max_depth
        )

def get_seeding_config(
    *
    , session : Session
) -> Seeding | None:
    return session.exec(
        select(Seeding).where(Seeding.id == 1)
    ).first()

def post_seeding_config_from_settings(
    *
    , session : Session
) -> Seeding:
    base = settings.seeding.model_dump()

    base['currid'] = session.exec(
        select(func.min(ItemDB.id))
    ).one() - 1 # we don't want to re-fetch an existing item

    seeding = Seeding(**base)

    return session_lifecycle(session, seeding)

def update_seeding(
    *
    , currid : int
    , last_execution_ts : Timestamp
) -> Seeding:
    with Session(DB.ENGINE) as session:
        original = get_seeding_config(session=session)

        original.sqlmodel_update({
            'currid': currid - 1
            , 'last_execution_ts': last_execution_ts
        })

        return session_lifecycle(session, original)

def init_seeding() -> Seeding:
    with Session(DB.ENGINE) as session:
        conf = get_seeding_config(session=session)

        if conf is None:
            return post_seeding_config_from_settings(
                session=session
            )

        # TODO: update seeding config from settings?

        return conf
