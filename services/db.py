from config import settings

import models.db

from sqlmodel import (
    Session, SQLModel, create_engine, select
)

class DB:

    ENGINE = create_engine(str(settings.db.postgres_db_uri))

    @classmethod
    def get_session(cls):
        with Session(cls.ENGINE) as session:
            yield session

    @classmethod
    def new_session(cls):
        """
        May be interesting to try and decorate this?
        """
        with Session(cls.ENGINE) as session:
            return session

    @classmethod
    def pass_session(cls, func, *args, **kw):
        with Session(cls.ENGINE) as session:
            try:
                result = func(*args, session=session, **kw)
            except Exception as e:
                # TODO: logger or return/raise exception
                print(e)

            return result

    @classmethod
    def read(cls, stmt):
        """
        TODO: assert stmt is a select
        """
        with Session(cls.ENGINE) as session:
            result = session.exec(stmt)
            ...

    @classmethod
    def create(cls, stmt):
        """
        TODO: assert stmt is a post
        """
        with Session(cls.ENGINE) as session:
            result = session.exec(stmt)
            ...

    @classmethod
    def update(cls, stmt):
        """
        TODO: assert stmt is an update
        """
        with Session(cls.ENGINE) as session:
            result = session.exec(stmt)
            ...

    @classmethod
    def create_all(cls):
        SQLModel.metadata.create_all(cls.ENGINE)
