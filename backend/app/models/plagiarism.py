"""
Plagiarism detection data models for the academic publication management system.
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, Float, ForeignKey, JSON
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pydantic import BaseModel, validator
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

Base = declarative_base()


class PlagiarismStatus(str, Enum):
    """Plagiarism scan status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    FLAGGED = "flagged"
    CLEARED = "cleared"


class PlagiarismSeverity(str, Enum):
    """Plagiarism severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PlagiarismDetectionEngine(str, Enum):
    """Plagiarism detection engines."""
    BERT_SEMANTIC = "bert_semantic"
    EDEN_AI = "eden_ai"
    GRAMMARLY = "grammarly"
    COPYLEAKS = "copyleaks"
    MANUAL = "manual"


class PlagiarismScan(Base):
    """Plagiarism scan database model."""
    __tablename__ = "plagiarism_scans"

    id = Column(Integer, primary_key=True, index=True)
    publication_id = Column(Integer, ForeignKey("publications.id"), nullable=False, index=True)

    # Scan metadata
    scan_date = Column(DateTime(timezone=True), server_default=func.now())
    engine = Column(String(50), nullable=False, index=True)
    scan_version = Column(String(50), nullable=True)

    # Results
    overall_similarity_score = Column(Float, nullable=False, index=True)
    max_similarity_score = Column(Float, nullable=False)
    severity_level = Column(String(20), nullable=False, index=True)

    # Status and processing
    status = Column(String(20), nullable=False, default=PlagiarismStatus.PENDING)
    processing_time_seconds = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)

    # Scan configuration
    check_against_internet = Column(Boolean, default=True)
    check_against_academic = Column(Boolean, default=True)
    check_against_internal = Column(Boolean, default=True)
    exclude_quotes = Column(Boolean, default=True)
    exclude_references = Column(Boolean, default=True)

    # Detailed results
    source_matches = Column(JSON, nullable=True)  # List of matching sources
    sentence_matches = Column(JSON, nullable=True)  # Sentence-level analysis
    paragraph_matches = Column(JSON, nullable=True)  # Paragraph-level analysis

    # Metadata
    scan_parameters = Column(JSON, nullable=True)
    engine_response = Column(JSON, nullable=True)  # Raw response from detection engine
    error_message = Column(Text, nullable=True)

    # Review and verification
    is_manually_reviewed = Column(Boolean, default=False)
    reviewer_id = Column(Integer, nullable=True)  # ID of reviewer
    review_date = Column(DateTime(timezone=True), nullable=True)
    review_notes = Column(Text, nullable=True)
    is_false_positive = Column(Boolean, default=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    publication = relationship("Publication", backref="plagiarism_scans")


class PlagiarismScanCreate(BaseModel):
    """Pydantic model for creating plagiarism scans."""
    publication_id: int
    engine: PlagiarismDetectionEngine
    scan_version: Optional[str] = None
    check_against_internet: bool = True
    check_against_academic: bool = True
    check_against_internal: bool = True
    exclude_quotes: bool = True
    exclude_references: bool = True
    scan_parameters: Optional[Dict[str, Any]] = None

    @validator('publication_id')
    def validate_publication_id(cls, v):
        if v <= 0:
            raise ValueError('Publication ID must be positive')
        return v


class PlagiarismScanUpdate(BaseModel):
    """Pydantic model for updating plagiarism scans."""
    overall_similarity_score: Optional[float] = None
    max_similarity_score: Optional[float] = None
    severity_level: Optional[PlagiarismSeverity] = None
    status: Optional[PlagiarismStatus] = None
    processing_time_seconds: Optional[float] = None
    confidence_score: Optional[float] = None
    source_matches: Optional[List[Dict[str, Any]]] = None
    sentence_matches: Optional[List[Dict[str, Any]]] = None
    paragraph_matches: Optional[List[Dict[str, Any]]] = None
    engine_response: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    is_manually_reviewed: Optional[bool] = None
    reviewer_id: Optional[int] = None
    review_date: Optional[datetime] = None
    review_notes: Optional[str] = None
    is_false_positive: Optional[bool] = None


class PlagiarismScanResponse(BaseModel):
    """Pydantic model for plagiarism scan response."""
    id: int
    publication_id: int
    scan_date: datetime
    engine: str
    scan_version: Optional[str] = None
    overall_similarity_score: float
    max_similarity_score: float
    severity_level: str
    status: str
    processing_time_seconds: Optional[float] = None
    confidence_score: Optional[float] = None
    check_against_internet: bool
    check_against_academic: bool
    check_against_internal: bool
    exclude_quotes: bool
    exclude_references: bool
    source_matches: Optional[List[Dict[str, Any]]] = None
    sentence_matches: Optional[List[Dict[str, Any]]] = None
    paragraph_matches: Optional[List[Dict[str, Any]]] = None
    scan_parameters: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    is_manually_reviewed: bool
    reviewer_id: Optional[int] = None
    review_date: Optional[datetime] = None
    review_notes: Optional[str] = None
    is_false_positive: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class PlagiarismSourceMatch(BaseModel):
    """Model for individual plagiarism source matches."""
    source_title: str
    source_url: Optional[str] = None
    source_author: Optional[str] = None
    similarity_score: float
    matched_text: str
    matched_length: int
    start_position: int
    end_position: int
    match_type: str  # exact, paraphrase, quote, etc.


class PlagiarismSentenceMatch(BaseModel):
    """Model for sentence-level plagiarism matches."""
    sentence_text: str
    sentence_index: int
    similarity_score: float
    matching_sources: List[PlagiarismSourceMatch]
    is_quote: bool = False
    is_reference: bool = False


class PlagiarismStats(BaseModel):
    """Plagiarism scan statistics model."""
    total_scans: int
    completed_scans: int
    pending_scans: int
    failed_scans: int
    flagged_scans: int
    cleared_scans: int
    average_similarity_score: float
    high_severity_count: int
    critical_severity_count: int
    scans_by_engine: Dict[str, int]
    scans_by_month: Dict[str, int]


class PlagiarismBatchConfig(BaseModel):
    """Configuration for batch plagiarism scanning."""
    publication_ids: List[int]
    engine: PlagiarismDetectionEngine
    priority: str  # high, medium, low
    batch_size: int = 10
    max_similarity_threshold: float = 0.3  # Only scan publications above this threshold
    exclude_previously_scanned: bool = True
    scan_configuration: Dict[str, Any] = {}

    @validator('publication_ids')
    def validate_publication_ids(cls, v):
        if not v:
            raise ValueError('At least one publication ID is required')
        if len(v) > 1000:
            raise ValueError('Cannot scan more than 1000 publications in a single batch')
        return v

    @validator('batch_size')
    def validate_batch_size(cls, v):
        if v < 1 or v > 50:
            raise ValueError('Batch size must be between 1 and 50')
        return v


class PlagiarismDetectionResult(BaseModel):
    """Model for individual plagiarism detection results."""
    publication_id: int
    engine: str
    overall_similarity: float
    max_similarity: float
    severity: PlagiarismSeverity
    processing_time: float
    source_matches: List[PlagiarismSourceMatch]
    confidence: float
    recommendations: List[str]


class PlagiarismReport(BaseModel):
    """Comprehensive plagiarism report model."""
    scan_id: int
    publication_title: str
    publication_id: int
    scan_date: datetime
    engine_used: str
    overall_similarity: float
    severity_level: str
    key_findings: List[str]
    source_analysis: Dict[str, Any]
    recommendations: List[str]
    manual_review_required: bool
    false_positive_indicators: List[str]
    confidence_score: float