"""
HackerNews post categorization using zero-shot and sentence transformers.
"""
import loader

from core.datatypes.sequence import Sequence
from core.utils.timer import Timer, ExecutionTimestamp

from analysis.db import (
    query
    , ItemFilter
    , IFilter
    , Queries
    , PresetIFilters
)

from analysis.sentiment.categories import (
    CATEGORY_DESCRIPTIONS, CATEGORIES
)

from analysis.sentiment.models.category import (
    TaggedPost, Category
)

from typing import TypeAlias

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from transformers import pipeline

import time

PostID: TypeAlias = int | str

class ZeroShotCategorizer:
    """
    Zero-shot classification using bart-large-mnli.

    Pros: No training needed, dynamic categories, good accuracy
    Cons: Slower inference, higher memory usage
    """

    def __init__(
        self
        , categories: list[str]
        , category_descriptions: dict[str, str] | None = None
    ):
        self.categories = categories
        self.descriptions = category_descriptions or {}
        self.classifier = pipeline(
            "zero-shot-classification"
            , model="facebook/bart-large-mnli"
            , device=-1  # CPU, use 0 for GPU
        )

    def categorize_batch(
        self
        , posts: list[str]
        , threshold: float = 0.5
    ) -> list[list[str]]:
        """Categorize posts with multi-label support."""
        results = []

        # Use descriptions if available, otherwise use category names
        candidate_labels = [
            self.descriptions.get(cat, cat)
            for cat in self.categories
        ]

        for post in posts:
            # Truncate very long posts
            text = post[:512] if len(post) > 512 else post

            output = self.classifier(
                text
                , candidate_labels
                , multi_label=True
            )

            # Filter by threshold
            categories = [
                self.categories[i]
                for i, score in enumerate(output['scores'])
                if score >= threshold
            ]
            results.append(categories)

        return results


class SentenceTransformerCategorizer:
    """
    Sentence transformer similarity-based categorization.

    Pros: Fast inference, flexible, interpretable
    Cons: May need tuning for optimal threshold
    """

    def __init__(
        self
        , categories: list[str]
        , category_descriptions: dict[str, str]
        , model_name: str = "all-MiniLM-L6-v2"
    ):
        self.categories = categories
        self.descriptions = category_descriptions
        self.model = SentenceTransformer(model_name)

        # Pre-compute category embeddings
        description_texts = [
            self.descriptions.get(cat, cat)
            for cat in self.categories
        ]
        self.category_embeddings = self.model.encode(
            description_texts
            , convert_to_tensor=False
            , show_progress_bar=False
        )

    def categorize_batch_base(
        self
        , posts: list[str]
        , threshold: float = 0.5
    ) -> list[list[str]]:
        """Categorize posts using cosine similarity."""
        # Encode all posts at once (efficient batching)
        post_embeddings = self.model.encode(
            posts
            , convert_to_tensor=False
            , show_progress_bar=True
            , batch_size=32
        )

        results = []
        for post_emb in post_embeddings:
            # Compute cosine similarity with all categories
            similarities = np.dot(
                self.category_embeddings
                , post_emb
            ) / (
                np.linalg.norm(self.category_embeddings, axis=1)
                * np.linalg.norm(post_emb)
            )

            # Filter by threshold
            categories = [
                self.categories[i]
                for i, sim in enumerate(similarities)
                if sim >= threshold
            ]
            results.append(categories)

        return results

    def categorize_batch_top_k(
        self
        , posts: list[str]
        , threshold: float = 0.36
        , min_categories: int = 1  # Always return at least 1
        , max_categories: int = 5  # Cap at 5
    ) -> list[list[str]]:
        """Categorize posts using cosine similarity."""
        post_embeddings = self.model.encode(
            posts
            , convert_to_tensor=False
            , show_progress_bar=True
            , batch_size=32
        )

        results = []
        for post_emb in post_embeddings:
            similarities = np.dot(
                self.category_embeddings
                , post_emb
            ) / (
                np.linalg.norm(self.category_embeddings, axis=1)
                * np.linalg.norm(post_emb)
            )

            # Get indices sorted by similarity (descending)
            sorted_indices = np.argsort(similarities)[::-1]

            # First, get all above threshold
            above_threshold = [
                self.categories[i]
                for i in sorted_indices
                if similarities[i] >= threshold
            ]

            # If we have enough, cap at max_categories
            if len(above_threshold) >= min_categories:
                categories = above_threshold[:max_categories]
            else:
                # Otherwise, take top min_categories
                categories = [
                    self.categories[i]
                    for i in sorted_indices[:min_categories]
                ]

            results.append(categories)

        return results

    def categorize_batch(
        self
        , posts: list[str]
        , ids:   list[int]
        , threshold: float = 0.4
        , min_categories: int = 1
        , max_categories: int = 3
        , confidence_levels: dict[str, float] | None = None
    ) -> Sequence[TaggedPost]:
        """
        Multi-level categorization with fallback strategies.
        """
        if confidence_levels is None:
            confidence_levels = {
                "high": threshold, "medium": 0.35, "low": 0.3
            }

        post_embeddings = self.model.encode(
            posts
            , convert_to_tensor=False
            #, show_progress_bar=True
            , batch_size=32
        )

        results = []
        for post_emb, post, id in zip(post_embeddings, posts, ids):
            base_tag_dict = {
                'id': id
                , 'title': post
            }

            similarities = np.dot(
                self.category_embeddings
                , post_emb
            ) / (
                np.linalg.norm(self.category_embeddings, axis=1)
                * np.linalg.norm(post_emb)
            )

            sorted_indices = np.argsort(similarities)[::-1]

            # Try each confidence level
            categories = []
            for level, thresh in sorted(
                confidence_levels.items()
                , key=lambda x: -x[1]
            ):
                categories = [
                    (self.categories[i], similarities[i], level)
                    for i in sorted_indices
                    if similarities[i] >= thresh
                ]
                if len(categories) >= min_categories:
                    break

            # Final fallback: just take top N
            if not categories:
                categories = [
                    (self.categories[i], similarities[i], 'low')
                    for i in sorted_indices[:min_categories]
                ]
            #else:
            #    categories = categories[:max_categories]

            base_tag_dict['tags'] = [
                Category(
                    label=label
                    , description=self.descriptions.get(label)
                    , confidence=conf
                    , level=level
                )
                for label, conf, level in categories
            ]

            results.append(
                TaggedPost(**base_tag_dict)
            )

        return Sequence(results)

# Example usage and benchmarking

def benchmark_categorizers(
    posts: list[str]
    , post_ids: list[PostID]
) -> pd.DataFrame:
    print("Initializing categorizers...")

    # Initialize both
    #zs_cat = ZeroShotCategorizer(
    #    CATEGORIES
    #    , CATEGORY_DESCRIPTIONS
    #)
    st_cat = SentenceTransformerCategorizer(
        CATEGORIES
        , CATEGORY_DESCRIPTIONS
        , model_name="all-mpnet-base-v2"
        #, model_name="mixedbread-ai/mxbai-embed-large-v1"
    )

    # Zero-shot classification
    #print("\n--- Zero-Shot Classification ---")
    #start = time.time()
    #zs_results = zs_cat.categorize_batch(posts, threshold=0.5)
    #zs_time = time.time() - start
    #print(f"Time: {zs_time:.2f}s ({zs_time/len(posts):.3f}s per post)")

    # Sentence transformers
    print("\n--- Sentence Transformers ---")
    start = time.time()
    st_results = st_cat.categorize_batch(posts, post_ids, threshold=0.36)
    st_time = time.time() - start
    print(f"Time: {st_time:.2f}s ({st_time/len(posts):.3f}s per post)")

    # Create comparison dataframe
    #for i, post_id in enumerate(post_ids):
    #    results.append({
    #        "post_id": post_id
    #        , "post_preview": posts[i][:80] + "..." if len(posts[i]) > 80 else posts[i]
    #        #, "zero_shot": ", ".join(zs_results[i]) or "none"
    #        , "sent_trans": st_results[i][0] or None
    #        , "confidence": st_results[i][1] or None
    #        #, "agreement": set(zs_results[i]) == set(st_results[i])
    #    })

    results = st_results.apply(
        lambda tp: tp.condensed_fmt()
    ).to_frame()

    return results

def categorize(
    filters : Sequence[ItemFilter]
    , limit=1000
) -> pd.DataFrame:
    df = Queries.Sentiment.Categories.get_posts(
        filters, limit
    )

    return benchmark_categorizers(
        df.title.to_list(), df.id.to_list()
    )

class PostTitleCategorizer:

    def __init__(
        self
        , filters=PresetIFilters.NORMAL_SCORES
        , threshold=0.36
        , limit=1000
    ):
        self.threshold = threshold
        self.filters   = filters
        self.limit     = limit

        self.model = SentenceTransformerCategorizer(
            CATEGORIES
            , CATEGORY_DESCRIPTIONS
            , model_name="all-mpnet-base-v2"
        )

    @Timer.timed
    def run(self, ret_type='df') -> tuple[
        Sequence[TaggedPost] | pd.DataFrame
        , ExecutionTimestamp
    ]:
        df = Queries.Sentiment.Categories.get_posts(
            self.filters, self.limit
        )

        results = self.model.categorize_batch(
            df.title.to_list(), df.id.to_list(), threshold=self.threshold
        )

        if ret_type == 'df':
            return results.apply(
                lambda tp: tp.condensed_fmt()
            ).to_frame()

        return results

# Example with sample HN posts
if __name__ == "__main__":
    #df = query("select id, title from items where type != 'comment' and score >= 100 order by id desc limit 100")

    #sample_posts = [
    #    "Show HN: FastHTML – A new Python web framework built on HTMX"
    #    , "Critical RCE vulnerability found in Log4j (CVE-2021-44228)"
    #    , "Understanding the 2024 inflation report and Fed policy changes"
    #    , "How we reduced our PostgreSQL query times by 90%"
    #    , "Anthropic raises $450M Series C led by Spark Capital"
    #    , "Introduction to transformer architectures and attention"
    #    , "Building a real-time collaborative editor with React and WebRTC"
    #]

    #sample_ids = list(range(len(sample_posts)))

    #sample_posts, sample_ids = (df.title.to_list(), df.id.to_list())

    #df = benchmark_categorizers(sample_posts, sample_ids)
    #df = categorize([
    #    IFilter.Neq(column='type', value='comment')
    #    , IFilter.Ge(column='score', value=150)
    #])
    cat = PostTitleCategorizer()
    df, execution_time = cat.run()

    print("\n--- Results ---")
    print(execution_time.model_dump())
    print(df)

    print("\n--- Category Statistics ---")
    print("\nTop 25 Categories:")
    print(
        df.explode('labels')
          .groupby('labels')
          .title.count()
          .sort_values(ascending=False)
          .iloc[:25]
    )
