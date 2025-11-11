"""
Publications management API endpoints.
Handles publication CRUD operations, search, and analytics.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
from typing import List, Dict, Any, Optional
import logging

from app.utils.database import get_db
from app.models.publication import Publication, PublicationCreate, PublicationUpdate, PublicationFilter
from app.models.plagiarism import PlagiarismScan
from app.models.faculty import Faculty
from app.services.database_crawler import DatabaseCrawler
from app.services.publication_processor import PublicationProcessor
from app.services.plagiarism_detector import PlagiarismDetector

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/", summary="Get all publications")
async def get_publications(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    faculty_id: Optional[int] = Query(None, description="Filter by faculty ID"),
    publication_type: Optional[str] = Query(None, description="Filter by publication type"),
    year_from: Optional[int] = Query(None, description="Filter publications from year"),
    year_to: Optional[int] = Query(None, description="Filter publications to year"),
    search: Optional[str] = Query(None, description="Search publications by title or author"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get a list of publications with optional filtering and search.
    """
    try:
        query = db.query(Publication)

        # Apply filters
        if faculty_id:
            query = query.filter(Publication.faculty_id == faculty_id)

        if publication_type:
            query = query.filter(Publication.publication_type == publication_type)

        if year_from:
            query = query.filter(Publication.publication_year >= year_from)

        if year_to:
            query = query.filter(Publication.publication_year <= year_to)

        if search:
            search_term = f"%{search}%"
            query = query.filter(
                or_(
                    Publication.title.ilike(search_term),
                    func.cast(Publication.authors, db.String).ilike(search_term),
                    Publication.venue_name.ilike(search_term)
                )
            )

        # Get total count
        total = query.count()

        # Apply pagination and ordering
        publications = query.order_by(Publication.publication_year.desc()).offset(skip).limit(limit).all()

        publication_data = []
        for pub in publications:
            pub_data = {
                "id": pub.id,
                "faculty_id": pub.faculty_id,
                "title": pub.title,
                "abstract": pub.abstract,
                "authors": pub.authors,
                "author_count": pub.author_count,
                "publication_year": pub.publication_year,
                "publication_type": pub.publication_type.value,
                "venue_name": pub.venue_name,
                "publisher": pub.publisher,
                "volume": pub.volume,
                "issue": pub.issue,
                "pages": pub.pages,
                "doi": pub.doi,
                "url": pub.url,
                "citation_count": pub.citation_count,
                "impact_factor": pub.impact_factor,
                "relevance_score": pub.relevance_score,
                "confidence_score": pub.confidence_score,
                "source_database": pub.source_database,
                "created_at": pub.created_at.isoformat(),
                "is_verified": pub.is_verified
            }
            publication_data.append(pub_data)

        return {
            "success": True,
            "data": {
                "publications": publication_data,
                "pagination": {
                    "total": total,
                    "skip": skip,
                    "limit": limit,
                    "pages": (total + limit - 1) // limit
                }
            }
        }

    except Exception as e:
        logger.error(f"Error retrieving publications: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving publications: {str(e)}"
        )


@router.get("/{publication_id}", summary="Get publication by ID")
async def get_publication_by_id(
    publication_id: int,
    include_plagiarism: bool = Query(False, description="Include plagiarism scan results"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get detailed information about a specific publication.
    """
    try:
        publication = db.query(Publication).filter(Publication.id == publication_id).first()

        if not publication:
            raise HTTPException(
                status_code=404,
                detail=f"Publication with ID {publication_id} not found"
            )

        # Get faculty information
        faculty = db.query(Faculty).filter(Faculty.id == publication.faculty_id).first()

        pub_data = {
            "id": publication.id,
            "faculty_id": publication.faculty_id,
            "faculty_name": faculty.name if faculty else "Unknown",
            "title": publication.title,
            "abstract": publication.abstract,
            "authors": publication.authors,
            "author_count": publication.author_count,
            "publication_year": publication.publication_year,
            "publication_type": publication.publication_type.value,
            "venue_name": publication.venue_name,
            "publisher": publication.publisher,
            "volume": publication.volume,
            "issue": publication.issue,
            "pages": publication.pages,
            "doi": publication.doi,
            "isbn": publication.isbn,
            "issn": publication.issn,
            "url": publication.url,
            "citation_count": publication.citation_count,
            "impact_factor": publication.impact_factor,
            "relevance_score": publication.relevance_score,
            "confidence_score": publication.confidence_score,
            "source_database": publication.source_database,
            "source_metadata": publication.source_metadata,
            "created_at": publication.created_at.isoformat(),
            "updated_at": publication.updated_at.isoformat() if publication.updated_at else None,
            "is_verified": publication.is_verified
        }

        # Include plagiarism scan if requested
        if include_plagiarism:
            plagiarism_scan = db.query(PlagiarismScan).filter(
                PlagiarismScan.publication_id == publication_id
            ).first()

            if plagiarism_scan:
                pub_data["plagiarism_scan"] = {
                    "id": plagiarism_scan.id,
                    "overall_similarity_score": plagiarism_scan.overall_similarity_score,
                    "max_similarity_score": plagiarism_scan.max_similarity_score,
                    "severity_level": plagiarism_scan.severity_level,
                    "status": plagiarism_scan.status,
                    "scan_date": plagiarism_scan.scan_date.isoformat(),
                    "engine": plagiarism_scan.engine,
                    "confidence_score": plagiarism_scan.confidence_score
                }

        return {
            "success": True,
            "data": pub_data
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving publication: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving publication: {str(e)}"
        )


@router.post("/", summary="Create new publication")
async def create_publication(
    publication_data: PublicationCreate,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Create a new publication record.
    """
    try:
        # Validate faculty exists
        faculty = db.query(Faculty).filter(Faculty.id == publication_data.faculty_id).first()
        if not faculty:
            raise HTTPException(
                status_code=404,
                detail=f"Faculty with ID {publication_data.faculty_id} not found"
            )

        # Check for duplicates based on DOI or title + year
        if publication_data.doi:
            existing = db.query(Publication).filter(Publication.doi == publication_data.doi).first()
            if existing:
                raise HTTPException(
                    status_code=400,
                    detail=f"Publication with DOI {publication_data.doi} already exists"
                )

        # Create new publication
        new_publication = Publication(**publication_data.dict())
        db.add(new_publication)
        db.commit()
        db.refresh(new_publication)

        return {
            "success": True,
            "message": "Publication created successfully",
            "data": {
                "id": new_publication.id,
                "title": new_publication.title,
                "faculty_name": faculty.name
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating publication: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error creating publication: {str(e)}"
        )


@router.put("/{publication_id}", summary="Update publication")
async def update_publication(
    publication_id: int,
    publication_data: PublicationUpdate,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Update an existing publication.
    """
    try:
        publication = db.query(Publication).filter(Publication.id == publication_id).first()

        if not publication:
            raise HTTPException(
                status_code=404,
                detail=f"Publication with ID {publication_id} not found"
            )

        # Update fields
        update_data = publication_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(publication, field, value)

        db.commit()
        db.refresh(publication)

        return {
            "success": True,
            "message": "Publication updated successfully",
            "data": {
                "id": publication.id,
                "title": publication.title
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating publication: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error updating publication: {str(e)}"
        )


@router.delete("/{publication_id}", summary="Delete publication")
async def delete_publication(
    publication_id: int,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Delete a publication record.
    """
    try:
        publication = db.query(Publication).filter(Publication.id == publication_id).first()

        if not publication:
            raise HTTPException(
                status_code=404,
                detail=f"Publication with ID {publication_id} not found"
            )

        # Delete plagiarism scans first
        db.query(PlagiarismScan).filter(PlagiarismScan.publication_id == publication_id).delete()

        # Delete publication
        db.delete(publication)
        db.commit()

        return {
            "success": True,
            "message": f"Publication '{publication.title}' deleted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting publication: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting publication: {str(e)}"
        )


@router.post("/{publication_id}/scan-plagiarism", summary="Scan publication for plagiarism")
async def scan_publication_plagiarism(
    publication_id: int,
    scan_config: Optional[Dict[str, Any]] = None,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Perform plagiarism detection on a specific publication.
    """
    try:
        publication = db.query(Publication).filter(Publication.id == publication_id).first()

        if not publication:
            raise HTTPException(
                status_code=404,
                detail=f"Publication with ID {publication_id} not found"
            )

        # Initialize plagiarism detector
        detector = PlagiarismDetector()

        # Perform scan
        scan = await detector.scan_publication(publication, scan_config)

        # Save scan results
        db.add(scan)
        db.commit()
        db.refresh(scan)

        return {
            "success": True,
            "message": "Plagiarism scan completed successfully",
            "data": {
                "scan_id": scan.id,
                "publication_id": publication.id,
                "publication_title": publication.title,
                "overall_similarity_score": scan.overall_similarity_score,
                "max_similarity_score": scan.max_similarity_score,
                "severity_level": scan.severity_level,
                "status": scan.status,
                "processing_time_seconds": scan.processing_time_seconds,
                "confidence_score": scan.confidence_score,
                "requires_manual_review": scan.status.value in ["flagged", "critical"]
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scanning publication for plagiarism: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error scanning publication for plagiarism: {str(e)}"
        )


@router.post("/batch-scan-plagiarism", summary="Batch scan publications for plagiarism")
async def batch_scan_plagiarism(
    faculty_ids: Optional[List[int]] = Query(None, description="Faculty IDs to scan"),
    year_from: Optional[int] = Query(None, description="Start year for scanning"),
    year_to: Optional[int] = Query(None, description="End year for scanning"),
    scan_config: Optional[Dict[str, Any]] = None,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Perform plagiarism detection on multiple publications.
    """
    try:
        # Build query for publications to scan
        query = db.query(Publication)

        if faculty_ids:
            query = query.filter(Publication.faculty_id.in_(faculty_ids))

        if year_from:
            query = query.filter(Publication.publication_year >= year_from)

        if year_to:
            query = query.filter(Publication.publication_year <= year_to)

        publications = query.all()

        if not publications:
            return {
                "success": True,
                "message": "No publications found for scanning",
                "data": {
                    "total_scanned": 0,
                    "scans_created": 0
                }
            }

        # Initialize plagiarism detector
        detector = PlagiarismDetector()

        # Perform batch scan
        scans = await detector.batch_scan_publications(publications, scan_config)

        # Save scan results
        saved_scans = []
        for scan in scans:
            db.add(scan)
            saved_scans.append(scan)

        db.commit()

        # Generate statistics
        stats = await detector.get_scan_statistics(saved_scans)

        return {
            "success": True,
            "message": f"Batch plagiarism scan completed for {len(publications)} publications",
            "data": {
                "total_publications": len(publications),
                "total_scans": len(saved_scans),
                "scan_statistics": stats,
                "scan_parameters": {
                    "faculty_ids": faculty_ids,
                    "year_from": year_from,
                    "year_to": year_to
                }
            }
        }

    except Exception as e:
        logger.error(f"Error in batch plagiarism scan: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error in batch plagiarism scan: {str(e)}"
        )


@router.get("/statistics/overview", summary="Get publication statistics")
async def get_publication_statistics(
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get overview statistics for all publications.
    """
    try:
        # Total publications
        total_publications = db.query(Publication).count()

        # Publications by type
        type_stats = db.query(
            Publication.publication_type,
            func.count(Publication.id).label('count')
        ).group_by(Publication.publication_type).all()

        publication_types = {pub_type.value: count for pub_type, count in type_stats}

        # Publications by year
        year_stats = db.query(
            Publication.publication_year,
            func.count(Publication.id).label('count')
        ).group_by(Publication.publication_year).order_by(Publication.publication_year.desc()).all()

        years = {str(year): count for year, count in year_stats}

        # Citation statistics
        citation_stats = db.query(
            func.sum(Publication.citation_count).label('total_citations'),
            func.avg(Publication.citation_count).label('avg_citations'),
            func.max(Publication.citation_count).label('max_citations')
        ).first()

        # Impact factor statistics
        impact_stats = db.query(
            func.avg(Publication.impact_factor).label('avg_impact_factor'),
            func.max(Publication.impact_factor).label('max_impact_factor')
        ).filter(Publication.impact_factor.isnot(None)).first()

        # Source database breakdown
        source_stats = db.query(
            Publication.source_database,
            func.count(Publication.id).label('count')
        ).group_by(Publication.source_database).all()

        sources = {source: count for source, count in source_stats}

        # Plagiarism scan statistics
        plagiarism_stats = db.query(
            func.count(PlagiarismScan.id).label('total_scans'),
            func.count(PlagiarismScan.id).filter(PlagiarismScan.status == 'flagged').label('flagged_scans'),
            func.avg(PlagiarismScan.overall_similarity_score).label('avg_similarity')
        ).first()

        return {
            "success": True,
            "data": {
                "overview": {
                    "total_publications": total_publications,
                    "publication_types": publication_types,
                    "years": years,
                    "sources": sources
                },
                "citations": {
                    "total": int(citation_stats.total_citations) if citation_stats.total_citations else 0,
                    "average": float(citation_stats.avg_citations) if citation_stats.avg_citations else 0.0,
                    "maximum": int(citation_stats.max_citations) if citation_stats.max_citations else 0
                },
                "impact_factors": {
                    "average": float(impact_stats.avg_impact_factor) if impact_stats.avg_impact_factor else 0.0,
                    "maximum": float(impact_stats.max_impact_factor) if impact_stats.max_impact_factor else 0.0
                },
                "plagiarism_scans": {
                    "total_scans": int(plagiarism_stats.total_scans) if plagiarism_stats.total_scans else 0,
                    "flagged_scans": int(plagiarism_stats.flagged_scans) if plagiarism_stats.flagged_scans else 0,
                    "average_similarity": float(plagiarism_stats.avg_similarity) if plagiarism_stats.avg_similarity else 0.0
                }
            }
        }

    except Exception as e:
        logger.error(f"Error retrieving publication statistics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving publication statistics: {str(e)}"
        )


@router.get("/search/advanced", summary="Advanced publication search")
async def advanced_search(
    title: Optional[str] = Query(None, description="Search in title"),
    author: Optional[str] = Query(None, description="Search in authors"),
    venue: Optional[str] = Query(None, description="Search in venue name"),
    abstract: Optional[str] = Query(None, description="Search in abstract"),
    year_from: Optional[int] = Query(None),
    year_to: Optional[int] = Query(None),
    publication_types: Optional[List[str]] = Query(None),
    has_doi: Optional[bool] = Query(None),
    min_citations: Optional[int] = Query(None),
    min_impact_factor: Optional[float] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Advanced search with multiple criteria.
    """
    try:
        query = db.query(Publication)

        # Apply all filters
        if title:
            query = query.filter(Publication.title.ilike(f"%{title}%"))

        if author:
            query = query.filter(func.cast(Publication.authors, db.String).ilike(f"%{author}%"))

        if venue:
            query = query.filter(Publication.venue_name.ilike(f"%{venue}%"))

        if abstract:
            query = query.filter(Publication.abstract.ilike(f"%{abstract}%"))

        if year_from:
            query = query.filter(Publication.publication_year >= year_from)

        if year_to:
            query = query.filter(Publication.publication_year <= year_to)

        if publication_types:
            query = query.filter(Publication.publication_type.in_(publication_types))

        if has_doi is not None:
            if has_doi:
                query = query.filter(Publication.doi.isnot(None))
            else:
                query = query.filter(Publication.doi.is_(None))

        if min_citations:
            query = query.filter(Publication.citation_count >= min_citations)

        if min_impact_factor:
            query = query.filter(Publication.impact_factor >= min_impact_factor)

        # Get total count
        total = query.count()

        # Apply pagination
        publications = query.order_by(Publication.publication_year.desc()).offset(skip).limit(limit).all()

        # Format results
        results = []
        for pub in publications:
            faculty = db.query(Faculty).filter(Faculty.id == pub.faculty_id).first()
            results.append({
                "id": pub.id,
                "title": pub.title,
                "authors": pub.authors,
                "publication_year": pub.publication_year,
                "publication_type": pub.publication_type.value,
                "venue_name": pub.venue_name,
                "doi": pub.doi,
                "citation_count": pub.citation_count,
                "impact_factor": pub.impact_factor,
                "faculty_name": faculty.name if faculty else "Unknown"
            })

        return {
            "success": True,
            "data": {
                "publications": results,
                "pagination": {
                    "total": total,
                    "skip": skip,
                    "limit": limit,
                    "pages": (total + limit - 1) // limit
                },
                "search_criteria": {
                    "title": title,
                    "author": author,
                    "venue": venue,
                    "abstract": abstract,
                    "year_from": year_from,
                    "year_to": year_to,
                    "publication_types": publication_types,
                    "has_doi": has_doi,
                    "min_citations": min_citations,
                    "min_impact_factor": min_impact_factor
                }
            }
        }

    except Exception as e:
        logger.error(f"Error in advanced search: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in advanced search: {str(e)}"
        )