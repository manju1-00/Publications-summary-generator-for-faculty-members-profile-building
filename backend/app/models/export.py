"""
Export job data models for the academic publication management system.
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


class ExportFormat(str, Enum):
    """Export format enumeration."""
    EXCEL = "excel"
    WORD = "word"
    PDF = "pdf"
    CSV = "csv"
    JSON = "json"


class ExportStatus(str, Enum):
    """Export job status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExportType(str, Enum):
    """Export type enumeration."""
    FACULTY_WISE = "faculty_wise"
    YEAR_WISE = "year_wise"
    TYPE_WISE = "type_wise"
    DEPARTMENT_WISE = "department_wise"
    CUSTOM_DURATION = "custom_duration"
    SUMMARY_DASHBOARD = "summary_dashboard"
    PLAGIARISM_REPORT = "plagiarism_report"


class ExportJob(Base):
    """Export job database model."""
    __tablename__ = "export_jobs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)

    # Export configuration
    export_type = Column(String(50), nullable=False, index=True)
    export_format = Column(String(20), nullable=False)
    title = Column(String(500), nullable=False)

    # Filter parameters
    faculty_ids = Column(JSON, nullable=True)  # List of faculty IDs
    departments = Column(JSON, nullable=True)  # List of departments
    publication_types = Column(JSON, nullable=True)  # List of publication types
    year_from = Column(Integer, nullable=True, index=True)
    year_to = Column(Integer, nullable=True, index=True)
    include_plagiarism_data = Column(Boolean, default=False)
    include_citations = Column(Boolean, default=True)
    include_impact_factors = Column(Boolean, default=True)

    # Export options
    citation_style = Column(String(50), default="APA")  # APA, MLA, IEEE, etc.
    language = Column(String(10), default="en")
    include_images = Column(Boolean, default=False)
    include_collaboration_network = Column(Boolean, default=False)

    # Processing information
    status = Column(String(20), nullable=False, default=ExportStatus.PENDING, index=True)
    progress_percentage = Column(Float, default=0.0)
    total_records = Column(Integer, default=0)
    processed_records = Column(Integer, default=0)

    # Timing
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    estimated_completion = Column(DateTime(timezone=True), nullable=True)

    # Results
    file_url = Column(String(1000), nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    download_count = Column(Integer, default=0)
    expires_at = Column(DateTime(timezone=True), nullable=True)

    # Metadata and error handling
    parameters = Column(JSON, nullable=True)  # Additional export parameters
    error_message = Column(Text, nullable=True)
    processing_log = Column(JSON, nullable=True)  # Processing steps log

    # Quality metrics
    record_count = Column(Integer, nullable=True)
    faculty_count = Column(Integer, nullable=True)
    department_count = Column(Integer, nullable=True)


class ExportJobCreate(BaseModel):
    """Pydantic model for creating export jobs."""
    user_id: int
    export_type: ExportType
    export_format: ExportFormat
    title: str
    faculty_ids: Optional[List[int]] = None
    departments: Optional[List[str]] = None
    publication_types: Optional[List[str]] = None
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    include_plagiarism_data: bool = False
    include_citations: bool = True
    include_impact_factors: bool = True
    citation_style: str = "APA"
    language: str = "en"
    include_images: bool = False
    include_collaboration_network: bool = False
    parameters: Optional[Dict[str, Any]] = None

    @validator('title')
    def validate_title(cls, v):
        if not v or len(v.strip()) < 3:
            raise ValueError('Export title must be at least 3 characters long')
        return v.strip()

    @validator('year_from', 'year_to')
    def validate_years(cls, v):
        if v and (v < 1900 or v > datetime.now().year + 2):
            raise ValueError('Year must be between 1900 and current year + 2')
        return v

    @validator('citation_style')
    def validate_citation_style(cls, v):
        allowed_styles = ["APA", "MLA", "IEEE", "Chicago", "Harvard", "Vancouver"]
        if v not in allowed_styles:
            raise ValueError(f'Citation style must be one of: {", ".join(allowed_styles)}')
        return v

    @validator('language')
    def validate_language(cls, v):
        allowed_languages = ["en", "es", "fr", "de", "pt", "zh", "ja", "ko"]
        if v not in allowed_languages:
            raise ValueError(f'Language must be one of: {", ".join(allowed_languages)}')
        return v


class ExportJobUpdate(BaseModel):
    """Pydantic model for updating export jobs."""
    status: Optional[ExportStatus] = None
    progress_percentage: Optional[float] = None
    total_records: Optional[int] = None
    processed_records: Optional[int] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    file_url: Optional[str] = None
    file_size_bytes: Optional[int] = None
    error_message: Optional[str] = None
    processing_log: Optional[List[Dict[str, Any]]] = None


class ExportJobResponse(BaseModel):
    """Pydantic model for export job response."""
    id: int
    user_id: int
    export_type: str
    export_format: str
    title: str
    faculty_ids: Optional[List[int]] = None
    departments: Optional[List[str]] = None
    publication_types: Optional[List[str]] = None
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    include_plagiarism_data: bool
    include_citations: bool
    include_impact_factors: bool
    citation_style: str
    language: str
    include_images: bool
    include_collaboration_network: bool
    status: str
    progress_percentage: float
    total_records: int
    processed_records: int
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    file_url: Optional[str] = None
    file_size_bytes: Optional[int] = None
    download_count: int
    expires_at: Optional[datetime] = None
    parameters: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    processing_log: Optional[List[Dict[str, Any]]] = None
    record_count: Optional[int] = None
    faculty_count: Optional[int] = None
    department_count: Optional[int] = None

    class Config:
        from_attributes = True


class ExportTemplate(BaseModel):
    """Export template model for predefined export configurations."""
    name: str
    description: str
    export_type: ExportType
    export_format: ExportFormat
    default_parameters: Dict[str, Any]
    is_system_template: bool = True


class ExportStats(BaseModel):
    """Export job statistics model."""
    total_exports: int
    completed_exports: int
    pending_exports: int
    failed_exports: int
    average_processing_time_minutes: float
    most_common_format: str
    most_common_type: str
    exports_by_month: Dict[str, int]
    exports_by_user: Dict[str, int]
    total_file_size_mb: float


class ExportProgress(BaseModel):
    """Export job progress model."""
    job_id: int
    status: ExportStatus
    progress_percentage: float
    current_step: str
    estimated_time_remaining_minutes: Optional[float] = None
    processed_records: int
    total_records: int
    error_message: Optional[str] = None


class ExportDownload(BaseModel):
    """Export download tracking model."""
    job_id: int
    download_date: datetime
    user_id: int
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class ExportQualityMetrics(BaseModel):
    """Export quality metrics model."""
    job_id: int
    record_count: int
    faculty_count: int
    department_count: int
    publication_types_count: Dict[str, int]
    years_covered: List[int]
    data_completeness_score: float
    formatting_quality_score: float
    has_all_requested_data: bool
    missing_data_summary: Dict[str, int]


class ExportFileMetadata(BaseModel):
    """Export file metadata model."""
    job_id: int
    file_name: str
    file_format: str
    file_size_bytes: int
    file_size_mb: float
    created_at: datetime
    checksum: Optional[str] = None
    contains_plagiarism_data: bool
    contains_images: bool
    contains_charts: bool
    sheet_count: Optional[int] = None  # For Excel files
    page_count: Optional[int] = None   # For Word/PDF files
    table_count: Optional[int] = None