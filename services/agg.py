from core.datatypes.sequence import Sequence

from services.inspector import Inspector
from models.base import (
    ItemT
    , Job
    , HackerT
    , ResponseError
    , StoryId
)

from typing import Callable

import multiprocessing as mp
import numpy as np

class Tasks:
    @staticmethod
    def _worker(
        ids: Sequence[StoryId]
    ) -> Sequence[ItemT]:
        return ids.apply(Inspector.get_item)

    @staticmethod
    def parallel(
        data: Sequence[StoryId]
        , max_procs: int = 12
        , batches: int = 1
    ) -> Sequence[ItemT]:
        if not data:
            return []

        # Split data into batches if batches > 1
        if data.length() > 1:
            batches = data.batch(batches)
        else:
            batches = [data]

        all_results = Sequence([])

        for batch in batches:
            # Further split batch into chunks for each process
            chunks = batch.batch(
                min(max_procs, batch.length())
            )

            with mp.Pool(
                processes=min(chunks.length(), max_procs)
            ) as pool:
                results = pool.map(Tasks._worker, chunks)

            all_results.extend(Sequence(results).flatten())

        return all_results

class Aggregator:

    @staticmethod
    def sagg(
        InspectorF : Callable
        , attr : str = 'stories'
    ) -> Sequence[ItemT]:
        # TODO: Inspector.get_updates will require some
        # attr shimming since its not just stories that we need to
        # pull from, also applies to pagg below
        return getattr(InspectorF(), attr).apply(
            Inspector.get_item
        )

    @staticmethod
    def pagg(
        InspectorF : Callable
        , attr : str = 'stories'
        , **kw
    ) -> Sequence[ItemT]:
        return Tasks.parallel(
            getattr(InspectorF(), attr)
            , **kw
        )

# ------------------ Testing Aggregators -------------------- #

class SAggregator:

    @staticmethod
    def get_jobs() -> Sequence[Job]:
        return Inspector.job_stories().stories.apply(
            Inspector.get_item
        )

class PAggregator:

    @staticmethod
    def get_jobs(**kw) -> Sequence[Job]:
        return Tasks.parallel(
            Inspector.job_stories().stories
            , **kw
        )
