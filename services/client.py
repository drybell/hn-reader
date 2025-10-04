from core.datatypes.sequence import Sequence

from models.base import (
    ItemT
    , Job
    , HackerT
    , ResponseError
    , StoryId
    , Updated
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

    @staticmethod
    def _do_work(InspectorF : Callable, **kw) -> Sequence[ItemT]:
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
    def updates() -> Updated:
        raise NotImplementedError(
            "Need to modify Aggregator.*agg attr handling before" \
            "calling this function or things will explode!"
        )
        return HNClient._do_work(
            Inspector.get_updates
        )
