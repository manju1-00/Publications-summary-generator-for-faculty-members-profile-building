"""
Publication data models for the academic publication management system.
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


class PublicationType(str, Enum):
    """Publication type enumeration."""
    JOURNAL = "journal"
    CONFERENCE = "conference"
    BOOK = "book"
    BOOK_CHAPTER = "book_chapter"
    WORKSHOP = "workshop"
    THESIS = "thesis"
    TECHNICAL_REPORT = "technical_report"
    PREPRINT = "preprint"
    OTHER = "other"


class PublicationStatus(str, Enum):
    """Publication status enumeration."""
    PUBLISHED = "published"
    IN_PRESS = "in_press"
    ACCEPTED = "accepted"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"


class Publication(Base):
    """Publication database model."""
    __tablename__ = "publications"

    id = Column(Integer, primary_key=True, index=True)
    faculty_id = Column(Integer, ForeignKey("faculty.id"), nullable=False, index=True)

    # Basic publication information
    title = Column(String(1000), nullable=False, index=True)
    normalized_title = Column(String(1000), nullable=False, index=True)
    abstract = Column(Text, nullable=True)
    publication_type = Column(String(50), nullable=False, index=True)

    # Venue information
    venue_name = Column(String(500), nullable=True, index=True)
    publisher = Column(String(500), nullable=True)
    volume = Column(String(50), nullable=True)
    issue = Column(String(50), nullable=True)
    pages = Column(String(100), nullable=True)

    # Identification and linking
    doi = Column(String(255), nullable=True, unique=True, index=True)
    isbn = Column(String(20), nullable=True)
    issn = Column(String(20), nullable=True)
    url = Column(String(1000), nullable=True)

    # Date information
    publication_year = Column(Integer, nullable=False, index=True)
    publication_date = Column(DateTime(timezone=True), nullable=True)

    # Author information
    authors = Column(JSON, nullable=True)  # List of author names
    author_count = Column(Integer, nullable=True)

    # Metrics and impact
    citation_count = Column(Integer, default=0)
    impact_factor = Column(Float, nullable=True)

    # Quality and relevance
    relevance_score = Column(Float, default=0.0)  # Relevance to faculty
    confidence_score = Column(Float, default=0.0)  # Confidence in categorization

    # Source information
    source_database = Column(String(100), nullable=False)  # DBLP, Crossref, etc.
    source_metadata = Column(JSON, nullable=True)  # Raw metadata from source

    # Processing information
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_verified = Column(Boolean, default=False)  # Verified by faculty

    # Relationships
    faculty = relationship("Faculty", backref="publications")
    plagiarism_scans = relationship("PlagiarismScan", backref="publication")


class PublicationCreate(BaseModel):
    """Pydantic model for creating publications."""
    faculty_id: int
    title: str
    abstract: Optional[str] = None
    publication_type: PublicationType
    venue_name: Optional[str] = None
    publisher: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None
    isbn: Optional[str] = None
    issn: Optional[str] = None
    url: Optional[str] = None
    publication_year: int
    publication_date: Optional[datetime] = None
    authors: Optional[List[str]] = None
    author_count: Optional[int] = None
    citation_count: Optional[int] = 0
    impact_factor: Optional[float] = None
    source_database: str
    source_metadata: Optional[Dict[str, Any]] = None

    @validator('title')
    def validate_title(cls, v):
        if not v or len(v.strip()) < 5:
            raise ValueError('Publication title must be at least 5 characters long')
        return v.strip()

    @validator('publication_year')
    def validate_year(cls, v):
        current_year = datetime.now().year
        if v < 1900 or v > current_year + 2:  # Allow future publications up to 2 years
            raise ValueError(f'Publication year must be between 1900 and {current_year + 2}')
        return v

    @validator('doi')
    def validate_doi(cls, v):
        if v and not (v.startswith('10.') and '/' in v):
            raise ValueError('Invalid DOI format')
        return v

    @validator('authors')
    def validate_authors(cls, v):
        if v and len(v) > 100:
            raise ValueError('Cannot have more than 100 authors')
        return v


class PublicationUpdate(BaseModel):
    """Pydantic model for updating publications."""
    title: Optional[str] = None
    abstract: Optional[str] = None
    publication_type: Optional[PublicationType] = None
    venue_name: Optional[str] = None
    publisher: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None
    isbn: Optional[str] = None
    issn: Optional[str] = None
    url: Optional[str] = None
    publication_year: Optional[int] = None
    publication_date: Optional[datetime] = None
    authors: Optional[List[str]] = None
    author_count: Optional[int] = None
    citation_count: Optional[int] = None
    impact_factor: Optional[float] = None
    relevance_score: Optional[float] = None
    confidence_score: Optional[float] = None
    is_verified: Optional[bool] = None


class PublicationResponse(BaseModel):
    """Pydantic model for publication response."""
    id: int
    faculty_id: int
    title: str
    abstract: Optional[str] = None
    publication_type: str
    venue_name: Optional[str] = None
    publisher: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None
    isbn: Optional[str] = None
    issn: Optional[str] = None
    url: Optional[str] = None
    publication_year: int
    publication_date: Optional[datetime] = None
    authors: Optional[List[str]] = None
    author_count: Optional[int] = None
    citation_count: int
    impact_factor: Optional[float] = None
    relevance_score: float
    confidence_score: float
    source_database: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    is_verified: bool

    class Config:
        from_attributes = True


class PublicationStats(BaseModel):
    """Publication statistics model."""
    total_publications: int
    journal_articles: int
    conference_papers: int
    books: int
    book_chapters: int
    other_publications: int
    publications_by_year: Dict[int, int]
    average_citations: float
    total_citations: int
    h_index: int


class PublicationFilter(BaseModel):
    """Model for publication filtering."""
    faculty_ids: Optional[List[int]] = None
    publication_types: Optional[List[PublicationType]] = None
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    venues: Optional[List[str]] = None
    min_citations: Optional[int] = None
    verified_only: Optional[bool] = False
    has_plagiarism_scan: Optional[bool] = None


class PublicationBatch(BaseModel):
    """Model for batch publication operations."""
    publications: List[PublicationCreate]
    total_count: int

    @validator('total_count')
    def validate_total_count(cls, v, values):
        if 'publications' in values and v != len(values['publications']):
            raise ValueError('Total count must match publications list length')
        return v


class PublicationDeduplication(BaseModel):
    """Model for publication deduplication results."""
    duplicates: List[Dict[str, Any]]
    unique_publications: List[Dict[str, Any]]
    confidence_threshold: float
    duplicates_removed: int