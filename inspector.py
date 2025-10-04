import requests

from models import (
    Item
    , User
    , Story
    , Comment
    , Job
    , ItemT
    , Ids
    , ItemWrapper
    , ItemPath
    , HackerT
    , ResponseError
    , StoryId
    , Updates
)

from pydantic import BaseModel, ValidationError
from typing import Callable, get_type_hints
from functools import wraps

import inspect
import types

def parse_model(
    _func: Callable | None
    , *
    , model_cls: type[HackerT] | None = None
):

    def decorator(func: Callable[..., requests.Response]):

        @wraps(func)
        def wrapper(*args, **kwargs) -> HackerT | StoryId | ResponseError:
            # Call the original function (e.g., Inspector.get(...))
            response = None

            try:
                response = func(*args, **kwargs)
            except Exception as e:
                return ResponseError(
                    message="Request Error"
                    , error=e
                    , status=response.status_code
                )

            # Ensure it's a requests.Response
            if not response or not isinstance(response, requests.Response):
                return ResponseError(
                    status=999
                    , message="Expected a Response object"
                    , raw=response
                )

            if response.status_code > 201:
                return ResponseError(
                    status=response.status_code
                    , message="Request failed"
                    , raw=response
                )

            # Get model from decorator arg, or infer from return annotation
            Model = model_cls
            if Model is None:
                try:
                    sig = inspect.signature(func)
                    return_type = get_type_hints(func).get("return")
                    if return_type:
                        if issubclass(return_type, BaseModel):
                            Model = return_type
                        elif isinstance(return_type, types.UnionType):
                            Model = ItemWrapper
                        else:
                            # TODO: catch improper casts
                            Model = StoryId
                except Exception as e:
                    return ResponseError(
                        error=e
                        , message=f"Failed to determine Model: {e}"
                    )

            try:
                if Model == Ids:
                    return Model(stories=response.json())
                elif Model == ItemWrapper:
                    return Model(item=response.json()).item
                elif Model != StoryId:
                    return Model(**response.json())

                return response.json()
            except (ValueError, ValidationError) as e:
                return ResponseError(
                    error=e
                    , status=response.status_code
                    , raw=response
                    , message="Failed to load model"
                )

        return wrapper

    if _func is not None:
        return decorator(_func)

    return decorator

class Inspector:
    URL_BASE = "https://hacker-news.firebaseio.com/v0/"

    @staticmethod
    def get(
        path : ItemPath
        , id : int | str | None = None
    ) -> requests.Response:
        return requests.get(
            Inspector._gen_url_path(path, id)
            , headers={"Accept": "application/json"}
        )

    @staticmethod
    def _gen_url_path(
        path : ItemPath
        , id : int | str | None = None
    ) -> str:
        base = Inspector.URL_BASE + path.value

        if not id:
            return base

        match path:
            case ItemPath.ITEM | ItemPath.USER:
                return base + str(id) + '.json'
            case _:
                raise ValueError(f"{path} does not support id argument!")

    @staticmethod
    @parse_model
    def get_item(id : int | str) -> ItemT:
        return Inspector.get(ItemPath.ITEM, id)

    @staticmethod
    @parse_model
    def get_user(id : str) -> ItemT:
        return Inspector.get(ItemPath.USER, id)

    @staticmethod
    @parse_model
    def get_maxitem_id() -> StoryId:
        return Inspector.get(ItemPath.MAXITEM)

    @staticmethod
    @parse_model
    def top_stories() -> Ids:
        return Inspector.get(ItemPath.TOPSTORIES)

    @staticmethod
    @parse_model
    def new_stories() -> Ids:
        return Inspector.get(ItemPath.NEWSTORIES)

    @staticmethod
    @parse_model
    def best_stories() -> Ids:
        return Inspector.get(ItemPath.BESTSTORIES)

    @staticmethod
    @parse_model
    def ask_stories() -> Ids:
        return Inspector.get(ItemPath.ASKSTORIES)

    @staticmethod
    @parse_model
    def show_stories() -> Ids:
        return Inspector.get(ItemPath.SHOWSTORIES)

    @staticmethod
    @parse_model
    def jobs_stories() -> Ids:
        return Inspector.get(ItemPath.JOBSTORIES)

    @staticmethod
    @parse_model
    def get_updates() -> Updates:
        return Inspector.get(ItemPath.UPDATES)

