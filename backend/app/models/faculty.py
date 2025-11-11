"""
Faculty data models for the academic publication management system.
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel, EmailStr, validator
from datetime import datetime
from typing import Optional, List

Base = declarative_base()


class Faculty(Base):
    """Faculty database model."""
    __tablename__ = "faculty"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    normalized_name = Column(String(255), nullable=False, index=True)
    department = Column(String(255), nullable=False, index=True)
    email = Column(String(255), nullable=False, unique=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_active = Column(Boolean, default=True)

    # Additional faculty information
    title = Column(String(100), nullable=True)  # Professor, Associate Professor, etc.
    specialization = Column(Text, nullable=True)
    research_interests = Column(Text, nullable=True)


class FacultyCreate(BaseModel):
    """Pydantic model for creating faculty."""
    name: str
    department: str
    email: EmailStr
    title: Optional[str] = None
    specialization: Optional[str] = None
    research_interests: Optional[str] = None

    @validator('name')
    def validate_name(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError('Faculty name must be at least 2 characters long')
        return v.strip()

    @validator('department')
    def validate_department(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError('Department must be at least 2 characters long')
        return v.strip()

    @validator('email')
    def validate_email(cls, v):
        if not v or '@' not in v:
            raise ValueError('Invalid email format')
        return v.lower().strip()


class FacultyUpdate(BaseModel):
    """Pydantic model for updating faculty."""
    name: Optional[str] = None
    department: Optional[str] = None
    email: Optional[EmailStr] = None
    title: Optional[str] = None
    specialization: Optional[str] = None
    research_interests: Optional[str] = None
    is_active: Optional[bool] = None


class FacultyResponse(BaseModel):
    """Pydantic model for faculty response."""
    id: int
    name: str
    department: str
    email: str
    title: Optional[str] = None
    specialization: Optional[str] = None
    research_interests: Optional[str] = None
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class FacultyBatchUpload(BaseModel):
    """Pydantic model for batch faculty upload."""
    faculty_list: List[FacultyCreate]
    total_count: int

    @validator('total_count')
    def validate_total_count(cls, v, values):
        if 'faculty_list' in values and v != len(values['faculty_list']):
            raise ValueError('Total count must match faculty list length')
        if v > 500:
            raise ValueError('Cannot upload more than 500 faculty members at once')
        return v


class FacultyStats(BaseModel):
    """Faculty statistics model."""
    total_faculty: int
    active_faculty: int
    departments_count: int
    publications_total: int
    avg_publications_per_faculty: float
    departments: List[str]


class FacultyNameVariation(BaseModel):
    """Model for faculty name variations for database queries."""
    original_name: str
    normalized_name: str
    variations: List[str]
    initials_version: str
    full_name: str