"""
Transformer-based sentiment and tone analysis for Hacker News comments.

Uses pre-trained transformer models (BERT, RoBERTa, DistilBERT) for
sentiment analysis without requiring API calls or LLM services.
"""
from core.datatypes.sequence import Sequence

from analysis.sentiment.models.comment import (
    ArgumentQuality
    , CommentSentiment
    , EmotionProfile
    , EmotionType
    , SentimentLabel
    , SentimentScore
    , TaggedComment
    , ThreadSentiment
    , ToneAnalysis
    , ToneType
)

import analysis.utils as utils

from models.base import (
    CommentThreadExpanded
)

from collections import Counter
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from transformers import pipeline

import numpy as np
import re
import warnings


class ModelConfig(BaseModel):
    """Configuration for transformer models."""
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    emotion_model: str = "j-hartmann/emotion-english-distilroberta-base"
    nli_model: str = "roberta-large-mnli"# or "microsoft/deberta-v3-base"
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    batch_size: int = 8
    max_length: int = 1024


class HNSentimentTransformer:
    """Transformer-based sentiment analyzer for Hacker News comments."""
    def __init__(self, config: ModelConfig | None = None) -> None:
        """
        Initialize the transformer-based sentiment analyzer.

        Args:
            config: Model configuration
        """
        self.config = config or ModelConfig()
        self._sentiment_pipeline = None
        self._emotion_pipeline = None
        self._nli_pipeline = None
        self._initialize_models()
        self._technical_lexicon = self._build_technical_lexicon()
        self._claim_patterns = self._build_claim_patterns()

    def _initialize_models(self) -> None:
        """Initialize transformer models and pipelines."""
        # Sentiment analysis pipeline
        self._sentiment_pipeline = pipeline(
            "sentiment-analysis"
            , model=self.config.sentiment_model
            , device=self.config.device
            , max_length=self.config.max_length
            , truncation=True
        )

        # Emotion classification pipeline
        self._emotion_pipeline = pipeline(
            "text-classification"
            , model=self.config.emotion_model
            , device=self.config.device
            , max_length=self.config.max_length
            , truncation=True
            , top_k=None  # Get all emotion scores
        )

        self._nli_pipeline = pipeline(
            "zero-shot-classification"
            , model=self.config.nli_model
            , device=self.config.device
        )

    def _build_claim_patterns(self) -> dict[str, Any]:
        """Build patterns for detecting claims and evidence types."""
        return {
            "claim_indicators": {
                "assertion": {
                    "is", "are", "will", "can", "should", "must"
                    , "always", "never", "all", "every", "none"
                }
                , "judgment": {
                    "better", "worse", "best", "worst", "superior"
                    , "inferior", "optimal", "suboptimal", "good", "bad"
                }
                , "causal": {
                    "causes", "leads to", "results in", "because of"
                    , "due to", "reason", "why", "how"
                }
            }
            , "evidence_types": {
                "empirical": {
                    "measured", "tested", "observed", "data shows"
                    , "benchmark", "profiled", "monitored", "logged"
                    , "experiment", "study", "found"
                }
                , "anecdotal": {
                    "in my experience", "i've seen", "i've found"
                    , "worked for me", "tried", "used"
                }
                , "authoritative": {
                    "according to", "research shows", "paper"
                    , "documentation", "specification", "standard"
                    , "rfc", "blog post", "article"
                }
                , "logical": {
                    "therefore", "thus", "hence", "follows that"
                    , "implies", "means that", "given that", "since"
                }
            }
            , "quantifiers": {
                "numbers", "percent", "%", "times", "fold"
                , "magnitude", "order of", "ms", "seconds", "bytes"
                , "mb", "gb", "million", "thousand"
            }
            , "hedging": {
                "might", "may", "could", "possibly", "perhaps"
                , "probably", "likely", "seems", "appears"
                , "suggests", "indicates"
            }
            , "counterargument": {
                "but", "however", "although", "while", "whereas"
                , "on the other hand", "alternatively", "conversely"
            }
        }

    def _build_technical_lexicon(self) -> dict[str, set[str]]:
        """Build lexicon for pattern-based features."""
        return {
            "technical": {
                "algorithm", "api", "architecture", "backend", "database"
                , "implementation", "framework", "performance", "optimization"
                , "scalability", "memory", "cpu", "latency", "throughput"
                , "function", "class", "method", "variable", "async"
                , "thread", "process", "container", "distributed"
            }
            , "evidence": {
                "because", "since", "therefore", "thus", "given"
                , "based on", "according to", "shows", "demonstrates"
                , "proves", "evidence", "data", "benchmark", "measured"
            }
            , "examples": {
                "example", "for instance", "such as", "like", "e.g."
                , "specifically", "consider", "suppose", "imagine"
            }
            , "controversy": {
                "but", "however", "actually", "wrong", "disagree"
                , "incorrect", "misleading", "false", "fallacy"
                , "biased", "unfair", "ridiculous", "nonsense"
                , "shit", "fuck", "ass", "bitch"
            }
            , "skeptical": {
                "doubt", "skeptical", "questionable", "dubious"
                , "suspicious", "uncertain", "allegedly", "supposedly"
                , "claim", "really", "sure"
            }
            , "constructive": {
                "suggest", "recommend", "consider", "might", "could"
                , "perhaps", "maybe", "alternative", "instead"
                , "improvement", "better"
            }
        }

    def _preprocess_text(self, text: str) -> tuple[str, list[str]]:
        if not text:
            return "", []

        clean = utils.Cleaners.strip(text)

        return clean, re.findall(r'\b\w+\b', clean.lower())

    def _calculate_subjectivity(self, text: str, words: list[str]) -> float:
        """Calculate subjectivity using linguistic patterns."""
        subjective_indicators = {
            "i think", "i believe", "in my opinion", "seems"
            , "appears", "probably", "maybe", "might", "could"
            , "should", "feel", "felt", "personal", "perspective"
        }

        text_lower = text.lower()
        matches = sum(
            1 for indicator in subjective_indicators
            if indicator in text_lower
        )

        # Also check for first person pronouns
        first_person = sum(
            1 for w in words if w in {"i", "my", "me", "mine"}
        )

        subjectivity = min(
            (matches * 0.15 + first_person * 0.1), 1.0
        )
        return subjectivity

    def _analyze_sentiment_transformer(
        self, text: str
    ) -> tuple[float, float, str]:
        """
        Use transformer model for sentiment analysis.

        Returns:
            (polarity, confidence, label)
        """
        if not self._sentiment_pipeline:
            raise RuntimeError("Sentiment pipeline not initialized")

        result = self._sentiment_pipeline(text)[0]
        label = result["label"].lower()
        confidence = result["score"]

        # Map label to polarity score
        # RoBERTa sentiment: negative (0), neutral (1), positive (2)
        if "negative" in label or label == "label_0":
            polarity = -confidence
        elif "positive" in label or label == "label_2":
            polarity = confidence
        else:  # neutral or label_1
            polarity = 0.0

        # Normalize label
        if "negative" in label or label == "label_0":
            normalized_label = "negative"
        elif "positive" in label or label == "label_2":
            normalized_label = "positive"
        else:
            normalized_label = "neutral"

        return polarity, confidence, normalized_label

    def _analyze_emotions_transformer(
        self, text: str
    ) -> tuple[EmotionType, dict[EmotionType, float], float]:
        """
        Use transformer model for emotion detection.

        Returns:
            (dominant_emotion, emotion_scores, intensity)
        """
        if not self._emotion_pipeline:
            raise RuntimeError("Emotion pipeline not initialized")

        results = self._emotion_pipeline(text)[0]

        # Map model labels to our EmotionType enum
        emotion_mapping = {
            "joy": EmotionType.JOY
            , "anger": EmotionType.ANGER
            , "surprise": EmotionType.EXCITEMENT
            , "sadness": EmotionType.CONCERN
            , "fear": EmotionType.CONCERN
            , "disgust": EmotionType.FRUSTRATION
            , "neutral": EmotionType.NEUTRAL
        }

        emotion_scores: dict[EmotionType, float] = {}
        for result in results:
            label = result["label"].lower()
            score = result["score"]

            # Map to our emotion types
            if label in emotion_mapping:
                emotion_type = emotion_mapping[label]
                emotion_scores[emotion_type] = (
                    emotion_scores.get(emotion_type, 0.0) + score
                )

        # Add pattern-based emotions
        text_lower = text.lower()
        if any(
            word in text_lower
            for word in ["interesting", "curious", "wonder", "how", "why"]
        ):
            emotion_scores[EmotionType.CURIOSITY] = (
                emotion_scores.get(EmotionType.CURIOSITY, 0.0) + 0.3
            )

        if any(
            word in text_lower
            for word in ["doubt", "skeptical", "questionable"]
        ):
            emotion_scores[EmotionType.SKEPTICISM] = (
                emotion_scores.get(EmotionType.SKEPTICISM, 0.0) + 0.3
            )

        # Normalize
        total = sum(emotion_scores.values())
        if total > 0:
            emotion_scores = {
                k: v / total for k, v in emotion_scores.items()
            }

        dominant = max(
            emotion_scores.items()
            , key=lambda x: x[1]
            , default=(EmotionType.NEUTRAL, 0.0)
        )

        intensity = dominant[1] if emotion_scores else 0.0

        return dominant[0], emotion_scores, intensity

    def _detect_tone(
        self, text: str, words: list[str]
    ) -> ToneAnalysis:
        """Detect tone using pattern matching and heuristics."""
        scores: dict[ToneType, float] = {}

        # Technical tone
        tech_count = sum(
            1 for w in words
            if w in self._technical_lexicon["technical"]
        )
        scores[ToneType.TECHNICAL] = min(
            tech_count / max(len(words) * 0.1, 1), 1.0
        )

        # Formal tone
        has_complex = len(text.split('.')) > 2
        avg_word_len = (
            sum(len(w) for w in words) / len(words) if words else 0
        )
        scores[ToneType.FORMAL] = min(
            (avg_word_len / 6.0) * (1.2 if has_complex else 1.0), 1.0
        )

        # Casual tone
        casual_markers = {
            'lol', 'yeah', 'nope', 'yep', 'haha', 'btw', 'tbh'
        }
        casual_count = sum(1 for w in words if w in casual_markers)
        scores[ToneType.CASUAL] = min(casual_count / 2.0, 1.0)

        # Controversial tone
        controversy_count = sum(
            1 for w in words
            if w in self._technical_lexicon["controversy"]
        )
        controversy_score = min(controversy_count / 3.0, 1.0)
        scores[ToneType.CONTROVERSIAL] = controversy_score

        # Educational tone
        educational_markers = {
            'explain', 'because', 'therefore', 'example'
            , 'specifically', 'essentially', 'basically', 'means'
        }
        edu_count = sum(
            1 for w in words if w in educational_markers
        )
        scores[ToneType.EDUCATIONAL] = min(edu_count / 2.0, 1.0)

        # Skeptical tone
        skep_count = sum(
            1 for w in words
            if w in self._technical_lexicon["skeptical"]
        )
        scores[ToneType.SKEPTICAL] = min(skep_count / 2.0, 1.0)

        # Enthusiastic tone
        enthu_markers = {
            'amazing', 'awesome', 'incredible', 'love', 'excited'
            , 'brilliant', 'fantastic', 'excellent'
        }
        enthu_count = sum(1 for w in words if w in enthu_markers)
        exclamations = text.count('!')
        scores[ToneType.ENTHUSIASTIC] = min(
            (enthu_count + exclamations * 0.5) / 2.0, 1.0
        )

        # Constructive tone
        constr_count = sum(
            1 for w in words
            if w in self._technical_lexicon["constructive"]
        )
        scores[ToneType.CONSTRUCTIVE] = min(constr_count / 2.0, 1.0)

        # Get top 2 tones
        sorted_tones = sorted(
            scores.items(), key=lambda x: x[1], reverse=True
        )
        primary_tone = sorted_tones[0][0]
        secondary_tone = (
            sorted_tones[1][0]
            if len(sorted_tones) > 1 and sorted_tones[1][1] > 0.2
            else None
        )

        return ToneAnalysis(
            primary_tone=primary_tone
            , secondary_tone=secondary_tone
            , tone_confidence=scores
            , is_controversial=controversy_score > 0.3
            , controversy_score=controversy_score
        )

    def _extract_claims_and_evidence(
        self, text: str
    ) -> tuple[list[str], list[tuple[str, str]]]:
        """
        Extract claims and their supporting evidence from text.

        Returns:
            (claims, evidence_list) where evidence_list is [(evidence, type)]
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        claims = []
        evidence = []

        for sentence in sentences:
            sentence_lower = sentence.lower()

            # Detect claims
            is_claim = False
            for claim_type, indicators in self._claim_patterns[
                "claim_indicators"
            ].items():
                if any(ind in sentence_lower for ind in indicators):
                    is_claim = True
                    break

            if is_claim:
                claims.append(sentence)

            # Detect evidence and classify type
            for evidence_type, indicators in self._claim_patterns[
                "evidence_types"
            ].items():
                if any(ind in sentence_lower for ind in indicators):
                    evidence.append((sentence, evidence_type))
                    break

        return claims, evidence

    def _calculate_coherence_score(
        self, text: str, sentences: list[str]
    ) -> float:
        """
        Calculate logical coherence using NLI model.

        Measures how well sentences follow from each other.
        """
        if not self._nli_pipeline or len(sentences) < 2:
            return 0.5  # Neutral score

        coherence_scores = []

        # Check consecutive sentence pairs
        for i in range(len(sentences) - 1):
            premise = sentences[i]
            hypothesis = sentences[i + 1]

            if not premise.strip() or not hypothesis.strip():
                continue

            try:
                result = self._nli_pipeline(
                    hypothesis
                    , candidate_labels=["follows from premise", "unrelated"]
                    , hypothesis_template="This statement: {}"
                )

                # Get entailment score
                if result["labels"][0] == "follows from premise":
                    coherence_scores.append(result["scores"][0])
                else:
                    coherence_scores.append(1.0 - result["scores"][0])

            except Exception as e:
                warnings.warn(f"Failed to run NLI Pipeline on `{text[:50]}...`: {e}")

        if not coherence_scores:
            return 0.5

        return float(np.mean(coherence_scores))

    def _assess_evidence_quality(
        self, evidence_list: list[tuple[str, str]]
    ) -> float:
        """
        Assess the quality of evidence provided.

        Empirical > Authoritative > Logical > Anecdotal
        """
        if not evidence_list:
            return 0.0

        weights = {
            "empirical": 1.0
            , "authoritative": 0.8
            , "logical": 0.6
            , "anecdotal": 0.4
        }

        scores = [weights[ev_type] for _, ev_type in evidence_list]
        return float(np.mean(scores))

    def _detect_quantitative_support(
        self, text: str
    ) -> tuple[bool, float]:
        """
        Detect if argument uses quantitative data.

        Returns:
            (has_numbers, specificity_score)
        """
        text_lower = text.lower()

        # Look for numbers and units
        has_numbers = bool(re.search(r'\b\d+', text))

        quantifier_count = sum(
            1 for q in self._claim_patterns["quantifiers"]
            if q in text_lower
        )

        specificity = min(
            (quantifier_count + (2 if has_numbers else 0)) / 3.0
            , 1.0
        )

        return has_numbers, specificity

    def _assess_reasoning_quality(
        self, text: str, claims: list[str], evidence_list: list[tuple[str, str]]
    ) -> float:
        """
        Assess overall reasoning quality.

        Good reasoning: claims backed by evidence, logical flow, specificity
        """
        if not claims:
            return 0.3  # No claims = low reasoning score

        # Check claim-to-evidence ratio
        claim_evidence_ratio = min(len(evidence_list) / len(claims), 1.0)

        # Check for hedging (appropriate uncertainty)
        text_lower = text.lower()
        hedging_count = sum(
            1 for hedge in self._claim_patterns["hedging"]
            if hedge in text_lower
        )
        appropriate_hedging = min(hedging_count / max(len(claims), 1), 0.3)

        # Check for counterarguments (shows nuanced thinking)
        counter_count = sum(
            1 for counter in self._claim_patterns["counterargument"]
            if counter in text_lower
        )
        considers_alternatives = min(counter_count / 2.0, 0.2)

        reasoning_score = (
            claim_evidence_ratio * 0.5
            + appropriate_hedging * 0.3
            + considers_alternatives * 0.2
        )

        return min(reasoning_score, 1.0)

    def _assess_argument_quality(
        self, text: str, words: list[str], skip: bool = False
    ) -> ArgumentQuality:
        """
        Assess the quality of arguments using advanced analysis.

        Combines claim extraction, evidence detection, logical coherence,
        and reasoning quality assessment.
        """
        # Extract claims and evidence
        if skip:
            return ArgumentQuality()

        claims, evidence_list = self._extract_claims_and_evidence(text)

        # Basic checks
        has_examples = any(
            marker in text.lower()
            for marker in self._technical_lexicon["examples"]
        )

        has_evidence = len(evidence_list) > 0

        # Quantitative support
        has_numbers, specificity = self._detect_quantitative_support(text)

        # Evidence quality
        evidence_quality = self._assess_evidence_quality(evidence_list)

        # Logical structure using NLI
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) > 1:
            coherence = self._calculate_coherence_score(text, sentences)
        else:
            # Fallback to pattern-based
            logical_connectors = {
                'because', 'therefore', 'thus', 'hence', 'since'
                , 'if', 'then', 'so', 'consequently'
            }
            logic_count = sum(
                1 for conn in logical_connectors
                if conn in text.lower()
            )
            coherence = min(logic_count / 3.0, 1.0)

        # Reasoning quality
        reasoning_quality = self._assess_reasoning_quality(
            text, claims, evidence_list
        )

        # Combine coherence and reasoning for logical structure
        logical_structure = (
            coherence * 0.6
            + reasoning_quality * 0.3
            + specificity * 0.1
        )

        # Technical depth
        tech_count = sum(
            1 for w in words
            if w in self._technical_lexicon["technical"]
        )
        technical_depth = min(
            tech_count / max(len(words) * 0.15, 1)
            , 1.0
        )

        # Boost technical depth if has evidence + quantitative data
        if has_evidence and has_numbers:
            technical_depth = min(technical_depth * 1.3, 1.0)

        # Constructiveness
        constr_count = sum(
            1 for w in words
            if w in self._technical_lexicon["constructive"]
        )
        destructive_markers = {
            'stupid', 'idiotic', 'ridiculous', 'nonsense', 'absurd'
        }
        destruc_count = sum(
            1 for w in words if w in destructive_markers
        )

        # Consider evidence in constructiveness
        base_constructiveness = max(
            (constr_count - destruc_count) / 2.0, 0.0
        )

        # Bonus for providing evidence and alternatives
        evidence_bonus = 0.2 if has_evidence else 0.0

        constructiveness = min(
            base_constructiveness + evidence_bonus, 1.0
        )

        return ArgumentQuality(
            has_evidence=has_evidence
            , has_examples=has_examples
            , logical_structure=logical_structure
            , technical_depth=technical_depth
            , constructiveness=constructiveness
        )

    def _assess_argument_quality_v1(
        self, text: str, words: list[str]
    ) -> ArgumentQuality:
        """Assess the quality of arguments in the comment."""
        text_lower = text.lower()

        # Check for evidence markers
        has_evidence = any(
            marker in text_lower
            for marker in self._technical_lexicon["evidence"]
        )

        # Check for examples
        has_examples = any(
            marker in text_lower
            for marker in self._technical_lexicon["examples"]
        )

        # Logical structure (presence of connectors)
        logical_connectors = {
            'because', 'therefore', 'thus', 'hence', 'since'
            , 'if', 'then', 'so', 'consequently', 'as a result'
        }
        logic_count = sum(
            1 for conn in logical_connectors if conn in text_lower
        )
        logical_structure = min(logic_count / 3.0, 1.0)

        # Technical depth
        tech_count = sum(
            1 for w in words
            if w in self._technical_lexicon["technical"]
        )
        technical_depth = min(tech_count / max(len(words) * 0.15, 1), 1.0)

        # Constructiveness
        constr_count = sum(
            1 for w in words
            if w in self._technical_lexicon["constructive"]
        )
        destructive_markers = {
            'stupid', 'idiotic', 'ridiculous', 'nonsense', 'absurd'
        }
        destruc_count = sum(
            1 for w in words if w in destructive_markers
        )
        constructiveness = min(
            max((constr_count - destruc_count) / 2.0, 0.0), 1.0
        )

        return ArgumentQuality(
            has_evidence=has_evidence
            , has_examples=has_examples
            , logical_structure=logical_structure
            , technical_depth=technical_depth
            , constructiveness=constructiveness
        )

    def analyze_comment(
        self
        , comment_id: int
        , text: str
        , author: str
        , metadata: dict[str, Any] | None = None
        , skip_argument: bool = False
    ) -> TaggedComment | None:
        """
        Analyze a single comment using transformer models.

        Args:
            comment_id: Unique identifier for the comment
            text: Comment text content
            metadata: Optional additional metadata

        Returns:
            Complete sentiment analysis results
        """
        if not text:
            return None

        clean_text, words = self._preprocess_text(text)

        # Transformer-based analysis
        polarity, confidence, label = self._analyze_sentiment_transformer(
            clean_text
        )
        subjectivity = self._calculate_subjectivity(text, words)

        sentiment = SentimentScore(
            polarity=polarity
            , confidence=confidence
            , subjectivity=subjectivity
            , label=label
        )

        # Emotion analysis
        dominant_emotion, emotion_scores, intensity = (
            self._analyze_emotions_transformer(clean_text)
        )
        emotions = EmotionProfile(
            dominant_emotion=dominant_emotion
            , emotion_scores=emotion_scores
            , emotional_intensity=intensity
        )

        # Pattern-based analysis
        tone = self._detect_tone(text, words)
        argument_quality = self._assess_argument_quality(
            text, words, skip=skip_argument
        )

        return TaggedComment(
            id=comment_id
            , raw=text
            , author=author
            , cleaned=clean_text
            , tokens=words
            , sentiment=CommentSentiment(
                comment_id=comment_id
                , sentiment=sentiment
                , tone=tone
                , emotions=emotions
                , argument_quality=argument_quality
                , word_count=len(words)
                , metadata=metadata or {}
            )
        )

    def analyze_thread(
        self
        , thread : CommentThreadExpanded | None = None
        , skip_argument : bool = False
    ) -> ThreadSentiment:
        """
        Analyze sentiment patterns across a comment thread.

        Args:
            thread : CommentThreadExpanded

        Returns:
            Aggregate thread sentiment analysis
        """
        if not thread or thread.children.empty():
            return ThreadSentiment(
                thread_id=thread.id
                , comment_count=0
                , avg_polarity=0.0
                , consensus_level=0.0
                , debate_quality=0.0
                , dominant_tones=[]
                , debate_detected=False
                , sentiment_variance=0.0
            )

        comments = Sequence([
            thread.parent
            , *thread.children
        ]).apply(
            lambda comment: self.analyze_comment(
                comment.id, comment.text, comment.by
                , skip_argument=skip_argument
            )
        )

        polarities = comments.sentiment.sentiment.polarity.to_list()
        avg_polarity = float(np.mean(polarities))
        sentiment_variance = float(np.var(polarities))

        # Consensus level (inverse of variance)
        consensus_level = max(1.0 - sentiment_variance * 2.0, 0.0)

        # Debate quality (based on argument quality metrics)
        avg_logic = np.mean(
            comments.sentiment.argument_quality.logical_structure.to_list()
        )
        avg_constructiveness = np.mean(
            comments.sentiment.argument_quality.constructiveness.to_list()
        )
        debate_quality = float((avg_logic + avg_constructiveness) / 2.0)

        # Detect debates
        controversy_scores = comments.sentiment.tone.controversy_score.to_list()

        avg_controversy = float(np.mean(controversy_scores))
        debate_detected = (
            sentiment_variance > 0.3 and avg_controversy > 0.3
        )

        # Dominant tones & emotions
        tone_counter: Counter[ToneType] = Counter()
        emotion_counter: Counter[EmotionType] = Counter(
            comments.sentiment.emotions.dominant_emotion
        )
        label_counter: Counter[SentimentLabel] = Counter(
            comments.sentiment.sentiment.label
        )

        for comment in comments:
            tone_counter[comment.sentiment.tone.primary_tone] += 1
            if comment.sentiment.tone.secondary_tone:
                tone_counter[comment.sentiment.tone.secondary_tone] += 0.5

        dominant_tones = [
            tone for tone, _ in tone_counter.most_common(3)
        ]

        dominant_emotions = [
            emotion for emotion, _ in emotion_counter.most_common(3)
        ]

        dominant_labels = [
            label for label, _ in label_counter.most_common(3)
        ]

        avg_emotion_intensity = np.mean(
            comments.sentiment.emotions.emotional_intensity.to_list()
        )

        return ThreadSentiment(
            thread_id=thread.id
            , comment_count=len(comments)
            , avg_polarity=avg_polarity
            , consensus_level=consensus_level
            , debate_quality=debate_quality
            , dominant_tones=dominant_tones
            , dominant_emotions=dominant_emotions
            , dominant_labels=dominant_labels
            , avg_controversy=avg_controversy
            , avg_emotional_intensity=avg_emotion_intensity
            , debate_detected=debate_detected
            , sentiment_variance=sentiment_variance
            , comments=comments
        )

    def batch_analyze(
        self, comments: list[tuple[int, str, str]]
    ) -> list[TaggedComment]:
        """
        Analyze multiple comments in batch.

        Args:
            comments: List of (comment_id, text) tuples

        Returns:
            List of sentiment analysis results
        """
        results = []
        for comment_id, text, author in comments:
            result = self.analyze_comment(comment_id, text, author)
            results.append(result)

        return results

# Convenience function for quick analysis
def analyze_hn_comment_transformer(
    comment_id: int
    , text: str
    , author: str
    , device: str = "cpu"
    , skip_argument: bool = False
) -> TaggedComment:
    """Quick transformer-based analysis of a single HN comment."""
    config = ModelConfig(device=device)
    analyzer = HNSentimentTransformer(config)

    return analyzer.analyze_comment(
        comment_id, text, author, skip_argument=skip_argument
    )
