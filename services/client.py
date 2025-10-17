from core.datatypes.sequence import Sequence

from models.base import (
    ItemT
    , Job
    , HackerT
    , ResponseError
    , StoryId
    , Updated
    , Comment
    , Ids
)

from services.inspector import Inspector
from services.agg import Aggregator

from typing import Callable

class HNClient:
    # TODO: settings?
    # TODO: max_procs of 12 and batches of 1 results in an ssl error
    # when attempting to run on 500 ids. need to identify a method
    # to guard against potential request failures or misconfigured
    # parameters (especially with parallel aggregation)
    AggF       = Aggregator.pagg
    max_procs  = 24
    batches    = 20
    expand_ids = False

    # TODO: probably a more elegant solution than doing this
    # manual delegation. for now it works, but in the future
    # this should be abstracted or re-done
    @classmethod
    def _call_inspector_then_process(
        cls
        , InspectorF : Callable
        , **kw
    ):
        # TODO: more dynamic aggregator comparison
        if cls.AggF == Aggregator.pagg:
            return cls.AggF(
                InspectorF
                , **kw
                , max_procs=cls.max_procs
                , batches=cls.batches
            )
        else:
            return cls.AggF(
                InspectorF
                , **kw
            )

    @classmethod
    def _process(cls, data : Sequence[StoryId], **kw):
        if cls.AggF == Aggregator.pagg:
            return Aggregator.pagg_only_data(
                data
                , **kw
                , max_procs=cls.max_procs
                , batches=cls.batches
            )
        else:
            return Aggregator.sagg_only_data(
                data
                , **kw
            )

    @classmethod
    def _do_work(
        cls
        , InspectorF : Callable | None = None
        , data     : Sequence[StoryId] | None = None
        , **kw
    ) -> Sequence[ItemT]| Ids | ResponseError:
        if InspectorF is not None:
            if cls.expand_ids:
                return cls._call_inspector_then_process(
                    InspectorF, **kw
                )
            else:
                return InspectorF()

        if data is not None:
            return cls._process(data, **kw)

        raise Exception(f"Inspector function or `data` Sequence required")

    @classmethod
    def new(cls) -> Sequence[ItemT] | ResponseError:
        return cls._do_work(
            Inspector.new_stories
        )

    @classmethod
    def best(cls) -> Sequence[ItemT] | ResponseError:
        return cls._do_work(
            Inspector.best_stories
        )

    @classmethod
    def askhn(cls) -> Sequence[ItemT] | ResponseError:
        return cls._do_work(
            Inspector.ask_stories
        )

    @classmethod
    def showhn(cls) -> Sequence[ItemT] | ResponseError:
        return cls._do_work(
            Inspector.show_stories
        )

    @classmethod
    def jobs(cls) -> Sequence[Job] | ResponseError:
        return cls._do_work(
            Inspector.job_stories
        )

    @classmethod
    def get(
        cls
        , id : int | str
        , retry_count : int | None = None
    ) -> ItemT | ResponseError:
        match id:
            case int():
                return Inspector.get_item(
                    id, retries=retry_count
                )
            case str():
                return Inspector.get_user(
                    id, retries=retry_count
                )
            case _:
                raise ValueError(f"`{id}` : {type(id)} != (int | str)!")

    @classmethod
    def expand(
        cls
        , item : ItemT | Sequence[StoryId]
        , attr='kids'
        , depth=1
    ) -> Sequence[Comment]:
        # TODO: depth
        match item:
            case Sequence():
                data = item
            case _:
                data = getattr(item, attr)

        if data is None or not data:
            return Sequence([])

        return cls._do_work(
            data=data
        )

    @classmethod
    def updates(cls) -> Updated:
        raise NotImplementedError(
            "Need to modify Aggregator.*agg attr handling before "
            "calling this function or things will explode!"
        )
        return cls._do_work(
            Inspector.get_updates
        )

class ParallelHNClient(HNClient):
    AggF = Aggregator.pagg
    expand_ids = True

class SequentialHNClient(HNClient):
    AggF = Aggregator.sagg
    expand_ids = True
