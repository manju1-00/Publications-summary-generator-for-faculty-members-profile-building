"""
Faculty management API endpoints.
Handles faculty CRUD operations and statistics.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import logging

from app.utils.database import get_db
from app.models.faculty import Faculty, FacultyCreate, FacultyUpdate, FacultyResponse, FacultyStats
from app.models.publication import Publication
from app.services.database_crawler import DatabaseCrawler
from app.services.publication_processor import PublicationProcessor

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/", summary="Get all faculty members")
async def get_all_faculty(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    department: Optional[str] = Query(None, description="Filter by department"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    search: Optional[str] = Query(None, description="Search faculty by name or email"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get a list of all faculty members with optional filtering.

    Supports pagination and filtering by department, active status, and search.
    """
    try:
        query = db.query(Faculty)

        # Apply filters
        if department:
            query = query.filter(Faculty.department.ilike(f"%{department}%"))

        if is_active is not None:
            query = query.filter(Faculty.is_active == is_active)

        if search:
            search_term = f"%{search}%"
            query = query.filter(
                (Faculty.name.ilike(search_term)) |
                (Faculty.email.ilike(search_term))
            )

        # Get total count
        total = query.count()

        # Apply pagination
        faculty_list = query.offset(skip).limit(limit).all()

        faculty_response = [FacultyResponse.from_orm(faculty) for faculty in faculty_list]

        return {
            "success": True,
            "data": {
                "faculty": faculty_response,
                "pagination": {
                    "total": total,
                    "skip": skip,
                    "limit": limit,
                    "pages": (total + limit - 1) // limit
                }
            }
        }

    except Exception as e:
        logger.error(f"Error retrieving faculty list: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving faculty list: {str(e)}"
        )


@router.get("/{faculty_id}", summary="Get faculty member by ID")
async def get_faculty_by_id(
    faculty_id: int,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get detailed information about a specific faculty member.
    """
    try:
        faculty = db.query(Faculty).filter(Faculty.id == faculty_id).first()

        if not faculty:
            raise HTTPException(
                status_code=404,
                detail=f"Faculty member with ID {faculty_id} not found"
            )

        faculty_response = FacultyResponse.from_orm(faculty)

        return {
            "success": True,
            "data": faculty_response
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving faculty member: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving faculty member: {str(e)}"
        )


@router.post("/", summary="Create new faculty member")
async def create_faculty(
    faculty_data: FacultyCreate,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Create a new faculty member.
    """
    try:
        # Check if email already exists
        existing_faculty = db.query(Faculty).filter(Faculty.email == faculty_data.email).first()
        if existing_faculty:
            raise HTTPException(
                status_code=400,
                detail=f"Faculty member with email {faculty_data.email} already exists"
            )

        # Create new faculty member
        new_faculty = Faculty(**faculty_data.dict())
        db.add(new_faculty)
        db.commit()
        db.refresh(new_faculty)

        faculty_response = FacultyResponse.from_orm(new_faculty)

        return {
            "success": True,
            "message": "Faculty member created successfully",
            "data": faculty_response
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating faculty member: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error creating faculty member: {str(e)}"
        )


@router.put("/{faculty_id}", summary="Update faculty member")
async def update_faculty(
    faculty_id: int,
    faculty_data: FacultyUpdate,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Update an existing faculty member.
    """
    try:
        faculty = db.query(Faculty).filter(Faculty.id == faculty_id).first()

        if not faculty:
            raise HTTPException(
                status_code=404,
                detail=f"Faculty member with ID {faculty_id} not found"
            )

        # Check if email is being changed and if it already exists
        if faculty_data.email and faculty_data.email != faculty.email:
            existing_faculty = db.query(Faculty).filter(Faculty.email == faculty_data.email).first()
            if existing_faculty:
                raise HTTPException(
                    status_code=400,
                    detail=f"Faculty member with email {faculty_data.email} already exists"
                )

        # Update fields
        update_data = faculty_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(faculty, field, value)

        db.commit()
        db.refresh(faculty)

        faculty_response = FacultyResponse.from_orm(faculty)

        return {
            "success": True,
            "message": "Faculty member updated successfully",
            "data": faculty_response
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating faculty member: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error updating faculty member: {str(e)}"
        )


@router.delete("/{faculty_id}", summary="Delete faculty member")
async def delete_faculty(
    faculty_id: int,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Delete a faculty member (soft delete by setting is_active to False).
    """
    try:
        faculty = db.query(Faculty).filter(Faculty.id == faculty_id).first()

        if not faculty:
            raise HTTPException(
                status_code=404,
                detail=f"Faculty member with ID {faculty_id} not found"
            )

        # Soft delete
        faculty.is_active = False
        db.commit()

        return {
            "success": True,
            "message": f"Faculty member {faculty.name} deactivated successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting faculty member: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting faculty member: {str(e)}"
        )


@router.get("/{faculty_id}/publications", summary="Get faculty publications")
async def get_faculty_publications(
    faculty_id: int,
    year_from: Optional[int] = Query(None, description="Filter publications from year"),
    year_to: Optional[int] = Query(None, description="Filter publications to year"),
    publication_type: Optional[str] = Query(None, description="Filter by publication type"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get all publications for a specific faculty member.
    """
    try:
        # Check if faculty exists
        faculty = db.query(Faculty).filter(Faculty.id == faculty_id).first()
        if not faculty:
            raise HTTPException(
                status_code=404,
                detail=f"Faculty member with ID {faculty_id} not found"
            )

        # Query publications
        query = db.query(Publication).filter(Publication.faculty_id == faculty_id)

        # Apply filters
        if year_from:
            query = query.filter(Publication.publication_year >= year_from)
        if year_to:
            query = query.filter(Publication.publication_year <= year_to)
        if publication_type:
            query = query.filter(Publication.publication_type == publication_type)

        # Get total count
        total = query.count()

        # Apply pagination
        publications = query.offset(skip).limit(limit).all()

        publication_data = []
        for pub in publications:
            pub_data = {
                "id": pub.id,
                "title": pub.title,
                "authors": pub.authors,
                "publication_year": pub.publication_year,
                "publication_type": pub.publication_type.value,
                "venue_name": pub.venue_name,
                "doi": pub.doi,
                "citation_count": pub.citation_count,
                "relevance_score": pub.relevance_score,
                "confidence_score": pub.confidence_score,
                "source_database": pub.source_database
            }
            publication_data.append(pub_data)

        return {
            "success": True,
            "data": {
                "faculty": faculty.name,
                "publications": publication_data,
                "pagination": {
                    "total": total,
                    "skip": skip,
                    "limit": limit,
                    "pages": (total + limit - 1) // limit
                },
                "filters": {
                    "year_from": year_from,
                    "year_to": year_to,
                    "publication_type": publication_type
                }
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving faculty publications: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving faculty publications: {str(e)}"
        )


@router.post("/{faculty_id}/crawl-publications", summary="Crawl publications for faculty")
async def crawl_faculty_publications(
    faculty_id: int,
    year_from: Optional[int] = Query(None, description="Start year for crawling"),
    year_to: Optional[int] = Query(None, description="End year for crawling"),
    databases: Optional[List[str]] = Query(None, description="Databases to crawl"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Crawl academic databases to find publications for a specific faculty member.
    """
    try:
        # Check if faculty exists
        faculty = db.query(Faculty).filter(Faculty.id == faculty_id).first()
        if not faculty:
            raise HTTPException(
                status_code=404,
                detail=f"Faculty member with ID {faculty_id} not found"
            )

        # Initialize crawler and processor
        async with DatabaseCrawler() as crawler:
            processor = PublicationProcessor()

            # Crawl publications
            faculty_names = [faculty.name]
            crawled_publications = await crawler.crawl_faculty_publications(
                faculty_names=faculty_names,
                year_from=year_from,
                year_to=year_to,
                databases=databases
            )

            # Process publications
            processed_publications, stats = await processor.process_publications(
                publications=crawled_publications,
                faculty_mapping={faculty.name: faculty.id}
            )

            # Save to database
            saved_count = 0
            for pub in processed_publications:
                # Check if publication already exists (avoid duplicates)
                existing = db.query(Publication).filter(
                    Publication.title == pub.title,
                    Publication.faculty_id == faculty_id
                ).first()

                if not existing:
                    new_pub = Publication(**pub.dict())
                    db.add(new_pub)
                    saved_count += 1

            db.commit()

            return {
                "success": True,
                "message": f"Successfully crawled and processed publications for {faculty.name}",
                "data": {
                    "faculty_id": faculty_id,
                    "faculty_name": faculty.name,
                    "total_found": len(crawled_publications),
                    "after_processing": len(processed_publications),
                    "new_publications_saved": saved_count,
                    "processing_statistics": stats,
                    "crawl_parameters": {
                        "year_from": year_from,
                        "year_to": year_to,
                        "databases": databases or ["dblp", "crossref", "openalex"]
                    }
                }
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error crawling faculty publications: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error crawling faculty publications: {str(e)}"
        )


@router.get("/statistics/overview", summary="Get faculty statistics")
async def get_faculty_statistics(
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get overview statistics for all faculty members.
    """
    try:
        total_faculty = db.query(Faculty).count()
        active_faculty = db.query(Faculty).filter(Faculty.is_active == True).count()

        # Get department breakdown
        departments = db.query(Faculty.department, db.func.count(Faculty.id)).group_by(Faculty.department).all()
        department_counts = {dept: count for dept, count in departments}

        # Get publications statistics
        total_publications = db.query(Publication).count()
        avg_publications = total_publications / total_faculty if total_faculty > 0 else 0

        # Get publications by type
        pub_types = db.query(
            Publication.publication_type, db.func.count(Publication.id)
        ).group_by(Publication.publication_type).all()
        publication_types = {pub_type.value: count for pub_type, count in pub_types}

        stats = FacultyStats(
            total_faculty=total_faculty,
            active_faculty=active_faculty,
            departments_count=len(department_counts),
            publications_total=total_publications,
            avg_publications_per_faculty=round(avg_publications, 2),
            departments=list(department_counts.keys())
        )

        return {
            "success": True,
            "data": {
                "overview": stats.dict(),
                "department_breakdown": department_counts,
                "publication_types": publication_types
            }
        }

    except Exception as e:
        logger.error(f"Error retrieving faculty statistics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving faculty statistics: {str(e)}"
        )


@router.get("/departments", summary="Get all departments")
async def get_departments(
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get a list of all departments with faculty counts.
    """
    try:
        departments = db.query(
            Faculty.department,
            db.func.count(Faculty.id).label('faculty_count'),
            db.func.count(Publication.id).label('publication_count')
        ).outerjoin(Publication).group_by(Faculty.department).all()

        department_list = []
        for dept, fac_count, pub_count in departments:
            department_list.append({
                "name": dept,
                "faculty_count": fac_count,
                "publication_count": pub_count or 0
            })

        return {
            "success": True,
            "data": {
                "departments": sorted(department_list, key=lambda x: x['name']),
                "total_departments": len(department_list)
            }
        }

    except Exception as e:
        logger.error(f"Error retrieving departments: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving departments: {str(e)}"
        )