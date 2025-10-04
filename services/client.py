from core.datatypes.sequence import Sequence

from models.base import (
    ItemT
    , Job
    , HackerT
    , ResponseError
    , StoryId
    , Updated
    , Comment
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
    AggF      = Aggregator.pagg
    max_procs = 24
    batches   = 20

    # TODO: probably a more elegant solution than doing this
    # manual delegation. for now it works, but in the future
    # this should be abstracted or re-done
    @staticmethod
    def _call_inspector_then_process(
        InspectorF : Callable
        , **kw
    ):
        # TODO: more dynamic aggregator comparison
        if HNClient.AggF == Aggregator.pagg:
            return HNClient.AggF(
                InspectorF
                , **kw
                , max_procs=HNClient.max_procs
                , batches=HNClient.batches
            )
        else:
            return HNClient.AggF(
                InspectorF
                , **kw
            )

    @staticmethod
    def _process(data : Sequence[StoryId], **kw):
        if HNClient.AggF == Aggregator.pagg:
            return Aggregator.pagg_only_data(
                data
                , **kw
                , max_procs=HNClient.max_procs
                , batches=HNClient.batches
            )
        else:
            return Aggregator.sagg_only_data(
                data
                , **kw
            )

    @staticmethod
    def _do_work(
        InspectorF : Callable | None = None
        , data     : Sequence[StoryId] | None = None
        , **kw
    ) -> Sequence[ItemT]:
        if InspectorF is not None:
            return HNClient._call_inspector_then_process(
                InspectorF, **kw
            )

        if data is not None:
            return HNClient._process(data, **kw)

        raise Exception(f"Inspector function or `data` Sequence required")

    @staticmethod
    def new() -> Sequence[ItemT]:
        return HNClient._do_work(
            Inspector.new_stories
        )

    @staticmethod
    def best() -> Sequence[ItemT]:
        return HNClient._do_work(
            Inspector.best_stories
        )

    @staticmethod
    def askhn() -> Sequence[ItemT]:
        return HNClient._do_work(
            Inspector.ask_stories
        )

    @staticmethod
    def showhn() -> Sequence[ItemT]:
        return HNClient._do_work(
            Inspector.show_stories
        )

    @staticmethod
    def jobs() -> Sequence[Job]:
        return HNClient._do_work(
            Inspector.job_stories
        )

    @staticmethod
    def expand(
        item : ItemT | Sequence[StoryId]
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

        return HNClient._do_work(
            data=data
        )

    @staticmethod
    def updates() -> Updated:
        raise NotImplementedError(
            "Need to modify Aggregator.*agg attr handling before" \
            "calling this function or things will explode!"
        )
        return HNClient._do_work(
            Inspector.get_updates
        )
