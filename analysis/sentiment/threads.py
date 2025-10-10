"""
Thread-level debate and controversy detection for Hacker News.

Analyzes entire comment threads to detect debates, opposing viewpoints,
controversy, and argument patterns using semantic similarity and
conversation structure analysis.
"""

from core.datatypes.sequence import Sequence

from analysis.sentiment.models.comment import (
    DebateIntensity
    , ControversyType
    , ArgumentPattern
    , OpposingViewpoints
    , ThreadDebateAnalysis
)

import analysis.utils as utils

from models.base import (
    CommentThreadExpanded
    , CommentThread
    , DeletedCommentThread
    , DeadCommentThread
    , FlaggedCommentThread
    , CommentThreadT
)

import re

import numpy as np

from dataclasses import dataclass

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

from typing import Any

@dataclass
class CommentNode:
    """Represents a comment in the thread structure."""

    comment_id: int
    text: str
    embedding: np.ndarray | None = None
    parent_id: int | None = None
    children: list[int] = None

    def __post_init__(self) -> None:
        if self.children is None:
            self.children = []

class ThreadDebateAnalyzer:
    """Analyzes debate and controversy patterns in comment threads."""

    def __init__(self, device: str = "cpu") -> None:
        """
        Initialize the thread debate analyzer.

        Args:
            device: Device to run models on
        """
        self.device = device
        self._embedding_model = None
        self._initialize_models()
        self._disagreement_patterns = self._build_disagreement_patterns()

    def _initialize_models(self) -> None:
        """Initialize sentence embedding models."""

        # Use lightweight model for efficiency
        self._embedding_model = SentenceTransformer(
            'all-mpnet-base-v2' # or "all-MiniLM-L6-v2"
            , device=self.device
        )

    def _build_disagreement_patterns(self) -> dict[str, Any]:
        """Build patterns for detecting disagreement and debate."""
        return {
            "direct_disagreement": {
                "no", "wrong", "incorrect", "disagree", "false"
                , "not true", "actually", "but", "however"
                , "that's not", "you're wrong", "this is wrong"
            }
            , "strong_disagreement": {
                "absolutely not", "completely wrong", "totally disagree"
                , "that's ridiculous", "nonsense", "absurd"
                , "you're missing", "fail to", "clearly wrong"
            }
            , "rebuttal_markers": {
                "on the contrary", "in fact", "actually"
                , "the reality is", "the truth is", "what really"
                , "let me correct", "that's misleading"
            }
            , "addressing_markers": {
                "you said", "you mentioned", "your point", "you're"
                , "as you", "you claim", "according to you"
            }
            , "escalation_markers": {
                "obviously", "clearly", "any fool", "common sense"
                , "even a child", "ridiculous", "pathetic"
                , "laughable", "idiotic", "stupid"
            }
            , "civil_debate": {
                "i understand", "fair point", "i see what you mean"
                , "that's interesting", "good question", "you're right"
                , "valid concern", "i agree that", "thanks for"
            }
        }

    def _extract_embeddings(
        self, comments: Sequence[CommentThreadT]
    ) -> tuple[dict[int, np.ndarray], list[str], list[int]]:
        """
        Generate semantic embeddings for all comments.

        Args:
            comments: List of (comment_id, text) tuples

        Returns:
            Dictionary mapping comment_id to embedding vector
        """
        if not self._embedding_model:
            raise RuntimeError("Embedding model not initialized")

        filtered = comments.where(
            lambda comment: (
                comment.__class__.__name__ not in [
                    'DeadCommentThread'
                    , 'FlaggedCommentThread'
                    , 'DeletedCommentThread'
                ]
            )
        )

        ids = filtered.id.to_list()
        texts = filtered.text.apply(
            utils.Cleaners.strip
        ).to_list()

        embeddings = self._embedding_model.encode(
            texts
            , convert_to_numpy=True
            , show_progress_bar=False
        )

        return (
            {
                comment_id: emb
                for comment_id, emb in zip(ids, embeddings)
            }
            , texts
            , ids
        )

    def _detect_viewpoint_camps(
        self, embeddings: dict[int, np.ndarray]
    ) -> OpposingViewpoints:
        """
        Cluster comments into opposing viewpoint camps.

        Uses semantic similarity to identify distinct positions.
        """
        if len(embeddings) < 2:
            return OpposingViewpoints(
                num_camps=1
                , camp_sizes=[len(embeddings)]
                , camp_representatives=list(embeddings.keys())[:1]
                , polarization_score=0.0
            )

        # Stack embeddings
        comment_ids = list(embeddings.keys())
        embedding_matrix = np.stack([
            embeddings[cid] for cid in comment_ids
        ])

        # Determine optimal number of clusters (2-4)
        optimal_k = min(max(2, len(embeddings) // 3), 4)

        # Cluster comments
        kmeans = KMeans(
            n_clusters=optimal_k
            , random_state=42
            , n_init=10
        )
        labels = kmeans.fit_predict(embedding_matrix)

        # Get camp sizes
        camp_sizes = [int(np.sum(labels == i)) for i in range(optimal_k)]

        # Get representative comment for each camp (closest to centroid)
        representatives = []
        for i in range(optimal_k):
            camp_mask = labels == i
            camp_embeddings = embedding_matrix[camp_mask]
            centroid = kmeans.cluster_centers_[i]

            # Find closest comment to centroid
            distances = cosine_similarity(
                camp_embeddings, centroid.reshape(1, -1)
            ).flatten()
            closest_idx = np.argmax(distances)

            # Map back to comment_id
            camp_comment_ids = [
                cid for cid, label in zip(comment_ids, labels)
                if label == i
            ]
            representatives.append(camp_comment_ids[closest_idx])

        # Calculate polarization (distance between camp centroids)
        if optimal_k > 1:
            centroid_distances = cosine_similarity(
                kmeans.cluster_centers_
            )
            # Get minimum similarity between camps (max polarization)
            np.fill_diagonal(centroid_distances, np.inf)
            min_similarity = np.min(centroid_distances)
            # Convert similarity to polarization (0 = same, 1 = opposite)
            polarization = 1.0 - (min_similarity + 1.0) / 2.0
        else:
            polarization = 0.0

        return OpposingViewpoints(
            num_camps=optimal_k
            , camp_sizes=camp_sizes
            , camp_representatives=representatives
            , polarization_score=float(polarization)
        )

    def _analyze_disagreement_patterns(
        self, texts: list[str]
    ) -> tuple[float, bool, bool]:
        """
        Analyze disagreement patterns in comments.

        Returns:
            (disagreement_ratio, escalation_detected, civility)
        """
        disagreeing_count = 0
        escalation_count = 0
        civil_count = 0

        for text in texts:
            text_lower = text.lower()

            # Check for disagreement
            has_disagreement = any(
                pattern in text_lower
                for pattern in (
                    self._disagreement_patterns["direct_disagreement"]
                    | self._disagreement_patterns["strong_disagreement"]
                    | self._disagreement_patterns["rebuttal_markers"]
                )
            )

            if has_disagreement:
                disagreeing_count += 1

            # Check for escalation
            has_escalation = any(
                pattern in text_lower
                for pattern in self._disagreement_patterns[
                    "escalation_markers"
                ]
            )

            if has_escalation:
                escalation_count += 1

            # Check for civil debate markers
            has_civility = any(
                pattern in text_lower
                for pattern in self._disagreement_patterns["civil_debate"]
            )

            if has_civility:
                civil_count += 1

        disagreement_ratio = disagreeing_count / len(texts)
        escalation_detected = escalation_count > 0
        civility_present = civil_count > 0

        return disagreement_ratio, escalation_detected, civility_present

    def _detect_back_and_forth(
        self
        , comment_ids : list[int]
        , texts: list[str]
        , embeddings: dict[int, np.ndarray]
    ) -> bool:
        """
        Detect if there's a back-and-forth debate pattern.

        Looks for alternating similar/dissimilar comments indicating
        people responding to each other.
        """
        if len(texts) < 3:
            return False

        # Calculate similarity between consecutive comments
        similarities = []

        for i in range(len(comment_ids) - 1):
            emb1 = embeddings[comment_ids[i]]
            emb2 = embeddings[comment_ids[i + 1]]
            sim = cosine_similarity(
                emb1.reshape(1, -1), emb2.reshape(1, -1)
            )[0, 0]
            similarities.append(sim)

        # Look for pattern: high-low-high-low (responding to each other)
        # or consistently low (different viewpoints alternating)
        if len(similarities) < 2:
            return False

        # Check if similarities alternate or are consistently low
        variance = np.var(similarities)
        mean_sim = np.mean(similarities)

        # High variance = alternating pattern
        # Low mean = different viewpoints
        back_and_forth = (variance > 0.05) or (mean_sim < 0.6)

        return back_and_forth

    def _extract_contentions(
        self, texts: list[str]
    ) -> list[str]:
        """
        Extract key points of contention from the thread.

        Finds sentences with disagreement markers.
        """
        contentions = []

        for text in texts:
            sentences = re.split(r'[.!?]+', text)

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                sentence_lower = sentence.lower()

                # Check if sentence contains disagreement
                has_disagreement = any(
                    pattern in sentence_lower
                    for pattern in (
                        self._disagreement_patterns["direct_disagreement"]
                        | self._disagreement_patterns["strong_disagreement"]
                        | self._disagreement_patterns["rebuttal_markers"]
                    )
                )

                if has_disagreement and len(sentence.split()) > 5:
                    contentions.append(sentence)

        # Return top 5 most substantial contentions
        contentions = sorted(contentions, key=len, reverse=True)[:5]
        return contentions

    def _calculate_controversy_score(
        self
        , disagreement_ratio: float
        , polarization: float
        , escalation: bool
        , num_camps: int
    ) -> float:
        """
        Calculate overall controversy score.

        Combines multiple signals into single metric.
        """
        base_score = (
            disagreement_ratio * 0.4
            + polarization * 0.4
            + (num_camps - 1) / 3.0 * 0.2
        )

        # Boost if escalation detected
        if escalation:
            base_score = min(base_score * 1.5, 1.0)

        return base_score

    def _classify_controversy_type(
        self
        , texts: list[str]
        , controversy_score: float
    ) -> ControversyType:
        """
        Classify the type of controversy.

        Analyzes language patterns to determine controversy nature.
        """
        if controversy_score < 0.2:
            return ControversyType.NONE

        # Check for personal attacks
        attack_patterns = {
            "you're stupid", "you're an idiot", "you don't understand"
            , "you're wrong", "you clearly", "you obviously"
        }

        has_attacks = any(
            any(pattern in text.lower() for pattern in attack_patterns)
            for text in texts
        )

        if has_attacks:
            return ControversyType.PERSONAL_ATTACK

        # Check for factual disagreement (numbers, data, citations)
        factual_patterns = {
            "according to", "data shows", "studies", "research"
            , "statistics", "measured", "benchmark", "tested"
        }

        has_factual = sum(
            1 for text in texts
            if any(pattern in text.lower() for pattern in factual_patterns)
        ) / len(texts)

        if has_factual > 0.3:
            return ControversyType.FACTUAL_DISAGREEMENT

        # Check for value-based language
        value_patterns = {
            "should", "must", "ought", "right", "wrong"
            , "moral", "ethical", "fair", "unfair", "freedom"
            , "privacy", "security"
        }

        has_values = sum(
            1 for text in texts
            if any(pattern in text.lower() for pattern in value_patterns)
        ) / len(texts)

        if has_values > 0.4:
            return ControversyType.VALUE_CLASH

        # Default to opinion clash
        return ControversyType.OPINION_CLASH

    def _calculate_debate_quality(
        self
        , texts: list[str]
        , civility_present: bool
        , escalation: bool
    ) -> float:
        """
        Assess the quality of debate.

        High quality = civil, evidence-based, constructive
        Low quality = personal attacks, no evidence, escalation
        """
        quality_score = 0.5  # Start neutral

        # Check for evidence
        evidence_markers = {
            "because", "evidence", "data", "research", "study"
            , "according to", "measured", "tested", "shows"
        }

        has_evidence = sum(
            1 for text in texts
            if any(marker in text.lower() for marker in evidence_markers)
        )
        evidence_ratio = has_evidence / len(texts)
        quality_score += evidence_ratio * 0.3

        # Civility bonus
        if civility_present:
            quality_score += 0.2

        # Escalation penalty
        if escalation:
            quality_score -= 0.3

        # Check for ad hominem
        ad_hominem = {
            "stupid", "idiot", "moron", "fool", "ignorant", "clueless"
        }

        has_ad_hominem = any(
            any(attack in text.lower() for attack in ad_hominem)
            for text in texts
        )

        if has_ad_hominem:
            quality_score -= 0.2

        return max(0.0, min(1.0, quality_score))

    def _determine_intensity(
        self
        , controversy_score: float
        , escalation: bool
        , civility_score: float
    ) -> DebateIntensity:
        """Determine debate intensity level."""
        if controversy_score < 0.2:
            return DebateIntensity.NONE

        if escalation or civility_score < 0.3:
            if controversy_score > 0.7:
                return DebateIntensity.HOSTILE
            return DebateIntensity.HEATED

        if controversy_score > 0.6:
            return DebateIntensity.MODERATE

        return DebateIntensity.MILD

    def analyze_thread(
        self
        , thread : CommentThreadExpanded | None = None
    ) -> ThreadDebateAnalysis:
        """
        Analyze a comment thread for debate and controversy.

        Args:
            thread: CommentThreadExpanded

        Returns:
            Complete debate analysis
        """
        if thread is None or thread.children.length() < 2:
            return ThreadDebateAnalysis(
                thread_id=thread_id
                , comment_count=thread.children.length()
                , debate_intensity=DebateIntensity.NONE
                , controversy_type=ControversyType.NONE
                , controversy_score=0.0
                , debate_quality=0.5
                , opposing_viewpoints=OpposingViewpoints(
                    num_camps=1
                    , camp_sizes=[thread.children.length()]
                    , camp_representatives=[thread.children.first().id]
                    , polarization_score=0.0
                )
                , disagreement_ratio=0.0
                , escalation_detected=False
                , back_and_forth_detected=False
                , civility_score=0.5
                , key_contentions=[]
            )

        # Extract semantic embeddings
        embeddings, texts, ids = self._extract_embeddings(
            thread.comments()
        )

        # Detect opposing viewpoints
        viewpoints = self._detect_viewpoint_camps(embeddings)

        # Analyze disagreement patterns
        disagreement_ratio, escalation, civility = (
            self._analyze_disagreement_patterns(texts)
        )

        # Calculate civility score
        civility_score = 0.8 if civility else 0.3
        if escalation:
            civility_score = min(civility_score, 0.2)

        # Detect back-and-forth
        back_and_forth = self._detect_back_and_forth(
            ids, texts, embeddings
        )

        # Calculate controversy
        controversy_score = self._calculate_controversy_score(
            disagreement_ratio
            , viewpoints.polarization_score
            , escalation
            , viewpoints.num_camps
        )

        # Classify controversy type
        controversy_type = self._classify_controversy_type(
            texts, controversy_score
        )

        # Calculate debate quality
        debate_quality = self._calculate_debate_quality(
            texts, civility, escalation
        )

        # Extract contentions
        contentions = self._extract_contentions(texts)

        # Determine intensity
        intensity = self._determine_intensity(
            controversy_score, escalation, civility_score
        )

        return ThreadDebateAnalysis(
            thread_id=thread.id
            , comment_count=thread.comments().length()
            , debate_intensity=intensity
            , controversy_type=controversy_type
            , controversy_score=controversy_score
            , debate_quality=debate_quality
            , opposing_viewpoints=viewpoints
            , disagreement_ratio=disagreement_ratio
            , escalation_detected=escalation
            , back_and_forth_detected=back_and_forth
            , civility_score=civility_score
            , key_contentions=contentions
        )
