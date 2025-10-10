from core.datatypes.sequence import Sequence
from pydantic import BaseModel, Field
from enum import StrEnum

from typing import Any

class ToneType(StrEnum):
    """Detected tone categories for HN comments."""

    TECHNICAL = "technical"
    FORMAL = "formal"
    CASUAL = "casual"
    CONTROVERSIAL = "controversial"
    EDUCATIONAL = "educational"
    SKEPTICAL = "skeptical"
    ENTHUSIASTIC = "enthusiastic"
    CONSTRUCTIVE = "constructive"


class EmotionType(StrEnum):
    """Emotional categories detected in comments."""
    JOY = "joy"
    ANGER = "anger"
    CURIOSITY = "curiosity"
    SKEPTICISM = "skepticism"
    EXCITEMENT = "excitement"
    FRUSTRATION = "frustration"
    CONCERN = "concern"
    NEUTRAL = "neutral"

class SentimentLabel(StrEnum):
    POSITIVE = 'positive'
    NEGATIVE = 'negative'
    NEUTRAL = 'neutral'

class SentimentScore(BaseModel):
    """Sentiment analysis results for a single comment."""
    polarity: float = Field(
        ..., ge=-1.0, le=1.0, description="Sentiment polarity"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Model confidence"
    )
    subjectivity: float = Field(
        ..., ge=0.0, le=1.0, description="Subjectivity score"
    )
    label: SentimentLabel = SentimentLabel.NEUTRAL

class ToneAnalysis(BaseModel):
    """Tone detection results."""
    primary_tone: ToneType
    secondary_tone: ToneType | None = None
    tone_confidence: dict[ToneType, float] = Field(
        default_factory=dict
    )
    is_controversial: bool = False
    controversy_score: float = Field(default=0.0, ge=0.0, le=1.0)


class EmotionProfile(BaseModel):
    """Emotional analysis of comment content."""
    dominant_emotion: EmotionType
    emotion_scores: dict[EmotionType, float] = Field(
        default_factory=dict
    )
    emotional_intensity: float = Field(
        default=0.0, ge=0.0, le=1.0
    )

class ArgumentQuality(BaseModel):
    """Assessment of argument quality in technical discussions."""
    has_evidence: bool = False
    has_examples: bool = False
    logical_structure: float = Field(
        default=0.0, ge=0.0, le=1.0
    )
    technical_depth: float = Field(
        default=0.0, ge=0.0, le=1.0
    )
    constructiveness: float = Field(
        default=0.0, ge=0.0, le=1.0
    )

class CommentSentiment(BaseModel):
    """Complete transformer-based sentiment analysis for a comment."""
    comment_id: int
    sentiment: SentimentScore
    tone: ToneAnalysis
    emotions: EmotionProfile
    argument_quality: ArgumentQuality
    word_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)

class TaggedComment(BaseModel):
    id        : int
    author    : str | None = None
    raw       : str
    cleaned   : str
    tokens    : Sequence[str]
    sentiment : CommentSentiment

    def to_condensed_fmt(self) -> dict:
        return {
            'id': self.id
            , 'author': self.author
            , 'cleaned': self.cleaned
            , 'tokens': self.tokens
            , **self.sentiment.sentiment.model_dump()
            , 'primary_tone': self.sentiment.tone.primary_tone
            , 'secondary_tone': self.sentiment.tone.secondary_tone
            , 'controversy_score': self.sentiment.tone.is_controversial
            , 'controversy_score': self.sentiment.tone.controversy_score
            , 'emotional_intensity': self.sentiment.emotions.emotional_intensity
            , 'primary_emotion': self.sentiment.emotions.dominant_emotion
            , 'word_count': self.sentiment.word_count
            , **self.sentiment.argument_quality.model_dump()
        }

class ThreadSentiment(BaseModel):
    """Aggregate sentiment analysis for a comment thread."""
    thread_id: str
    comment_count: int
    avg_polarity: float
    consensus_level: float = Field(
        ge=0.0, le=1.0
        , description="Agreement level in thread"
    )
    debate_quality: float = Field(
        ge=0.0, le=1.0
    )
    dominant_tones: Sequence[ToneType]
    dominant_emotions: Sequence[EmotionType]
    dominant_labels: Sequence[SentimentLabel]
    avg_controversy: float
    avg_emotional_intensity: float
    debate_detected: bool
    sentiment_variance: float
    comments : Sequence[TaggedComment]

    def to_condensed_fmt(self) -> dict:
        base = self.model_dump(exclude_none=True)
        base.pop('comments', None)
        return base

    def comment_df(self):
        return self.comments.apply(
            lambda x: x.to_condensed_fmt()
        ).to_frame()


class DebateIntensity(StrEnum):
    """Level of debate intensity in a thread."""

    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    HEATED = "heated"
    HOSTILE = "hostile"

class ControversyType(StrEnum):
    """Type of controversy detected."""

    NONE = "none"
    FACTUAL_DISAGREEMENT = "factual_disagreement"
    OPINION_CLASH = "opinion_clash"
    INTERPRETATION_CONFLICT = "interpretation_conflict"
    VALUE_CLASH = "value_clash"
    PERSONAL_ATTACK = "personal_attack"

class ArgumentPattern(BaseModel):
    """Detected argument patterns in comments."""

    has_claim: bool = False
    has_rebuttal: bool = False
    has_evidence: bool = False
    addresses_previous: bool = False
    introduces_new_point: bool = False

class OpposingViewpoints(BaseModel):
    """Detected opposing viewpoints in thread."""

    num_camps: int = Field(
        ge=1, description="Number of distinct viewpoint camps"
    )
    camp_sizes: list[int] = Field(default_factory=list)
    camp_representatives: list[int] = Field(
        default_factory=list
        , description="Comment IDs representing each camp"
    )
    polarization_score: float = Field(
        ge=0.0, le=1.0
        , description="How polarized the viewpoints are"
    )

class ThreadDebateAnalysis(BaseModel):
    """Complete debate analysis for a comment thread."""
    thread_id: str
    comment_count: int
    debate_intensity: DebateIntensity
    controversy_type: ControversyType
    controversy_score: float = Field(ge=0.0, le=1.0)
    debate_quality: float = Field(
        ge=0.0, le=1.0
        , description="Quality of argumentation"
    )
    opposing_viewpoints: OpposingViewpoints
    disagreement_ratio: float = Field(
        ge=0.0, le=1.0
        , description="Ratio of disagreeing comments"
    )
    escalation_detected: bool = False
    back_and_forth_detected: bool = False
    civility_score: float = Field(
        ge=0.0, le=1.0
        , description="How civil the discussion is"
    )
    key_contentions: Sequence[str]



