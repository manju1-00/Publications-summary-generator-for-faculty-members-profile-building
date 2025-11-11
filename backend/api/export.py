"""
Export API endpoints for generating reports in various formats.
Handles export job management and report generation.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import logging
import uuid
from datetime import datetime

from app.utils.database import get_db
from app.models.export import ExportJob, ExportJobCreate, ExportJobUpdate, ExportStatus
from app.models.publication import Publication
from app.models.faculty import Faculty
from app.models.plagiarism import PlagiarismScan
from app.services.export_generator import ExportGenerator

router = APIRouter()
logger = logging.getLogger(__name__)

# Global dictionary to store background export jobs (in production, use Redis or database)
active_exports = {}


@router.post("/jobs", summary="Create new export job")
async def create_export_job(
    export_data: ExportJobCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Create a new export job for generating reports.

    Export types:
    - faculty_wise: Separate sections for each faculty member
    - year_wise: Organize publications by year
    - type_wise: Separate by publication type (journal, conference, etc.)
    - department_wise: Group by department
    - custom_duration: Publications within specified date range
    - summary_dashboard: Overview statistics and summaries
    - plagiarism_report: Include plagiarism scan results
    """
    try:
        # Create export job record
        export_job = ExportJob(
            user_id=export_data.user_id,
            export_type=export_data.export_type,
            export_format=export_data.export_format,
            title=export_data.title,
            faculty_ids=export_data.faculty_ids,
            departments=export_data.departments,
            publication_types=export_data.publication_types,
            year_from=export_data.year_from,
            year_to=export_data.year_to,
            include_plagiarism_data=export_data.include_plagiarism_data,
            include_citations=export_data.include_citations,
            include_impact_factors=export_data.include_impact_factors,
            citation_style=export_data.citation_style,
            language=export_data.language,
            include_images=export_data.include_images,
            include_collaboration_network=export_data.include_collaboration_network,
            parameters=export_data.parameters
        )

        db.add(export_job)
        db.commit()
        db.refresh(export_job)

        # Start background processing
        job_id = export_job.id
        background_tasks.add_task(
            process_export_job,
            job_id,
            db
        )

        return {
            "success": True,
            "message": "Export job created successfully",
            "data": {
                "job_id": job_id,
                "title": export_job.title,
                "export_type": export_job.export_type.value,
                "export_format": export_job.export_format.value,
                "status": export_job.status.value,
                "created_at": export_job.created_at.isoformat(),
                "estimated_completion": None
            }
        }

    except Exception as e:
        logger.error(f"Error creating export job: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error creating export job: {str(e)}"
        )


@router.get("/jobs/{job_id}", summary="Get export job status")
async def get_export_job_status(
    job_id: int,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get the status and progress of an export job.
    """
    try:
        export_job = db.query(ExportJob).filter(ExportJob.id == job_id).first()

        if not export_job:
            raise HTTPException(
                status_code=404,
                detail=f"Export job with ID {job_id} not found"
            )

        response_data = {
            "job_id": export_job.id,
            "title": export_job.title,
            "export_type": export_job.export_type.value,
            "export_format": export_job.export_format.value,
            "status": export_job.status.value,
            "progress_percentage": export_job.progress_percentage,
            "total_records": export_job.total_records,
            "processed_records": export_job.processed_records,
            "created_at": export_job.created_at.isoformat(),
            "started_at": export_job.started_at.isoformat() if export_job.started_at else None,
            "completed_at": export_job.completed_at.isoformat() if export_job.completed_at else None,
            "estimated_completion": export_job.estimated_completion.isoformat() if export_job.estimated_completion else None
        }

        # Add file URL if completed
        if export_job.status == ExportStatus.COMPLETED and export_job.file_url:
            response_data["file_url"] = export_job.file_url
            response_data["file_size_bytes"] = export_job.file_size_bytes
            response_data["download_count"] = export_job.download_count

        # Add error message if failed
        if export_job.status == ExportStatus.FAILED:
            response_data["error_message"] = export_job.error_message

        return {
            "success": True,
            "data": response_data
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving export job status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving export job status: {str(e)}"
        )


@router.get("/jobs", summary="List export jobs")
async def list_export_jobs(
    user_id: Optional[int] = Query(None, description="Filter by user ID"),
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get a list of export jobs with optional filtering.
    """
    try:
        query = db.query(ExportJob)

        if user_id:
            query = query.filter(ExportJob.user_id == user_id)

        if status_filter:
            query = query.filter(ExportJob.status == status_filter)

        total = query.count()
        jobs = query.order_by(ExportJob.created_at.desc()).offset(skip).limit(limit).all()

        job_list = []
        for job in jobs:
            job_data = {
                "job_id": job.id,
                "title": job.title,
                "export_type": job.export_type.value,
                "export_format": job.export_format.value,
                "status": job.status.value,
                "progress_percentage": job.progress_percentage,
                "created_at": job.created_at.isoformat(),
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "file_size_bytes": job.file_size_bytes
            }
            job_list.append(job_data)

        return {
            "success": True,
            "data": {
                "jobs": job_list,
                "pagination": {
                    "total": total,
                    "skip": skip,
                    "limit": limit,
                    "pages": (total + limit - 1) // limit
                }
            }
        }

    except Exception as e:
        logger.error(f"Error listing export jobs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing export jobs: {str(e)}"
        )


@router.delete("/jobs/{job_id}", summary="Cancel/delete export job")
async def delete_export_job(
    job_id: int,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Cancel an active export job or delete a completed job.
    """
    try:
        export_job = db.query(ExportJob).filter(ExportJob.id == job_id).first()

        if not export_job:
            raise HTTPException(
                status_code=404,
                detail=f"Export job with ID {job_id} not found"
            )

        if export_job.status in [ExportStatus.PENDING, ExportStatus.IN_PROGRESS]:
            # Cancel the job
            export_job.status = ExportStatus.CANCELLED
            db.commit()
            message = f"Export job {job_id} cancelled successfully"
        else:
            # Delete the job
            db.delete(export_job)
            db.commit()
            message = f"Export job {job_id} deleted successfully"

        return {
            "success": True,
            "message": message
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting export job: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting export job: {str(e)}"
        )


@router.get("/jobs/{job_id}/download", summary="Download exported file")
async def download_exported_file(
    job_id: int,
    db: Session = Depends(get_db)
) -> StreamingResponse:
    """
    Download the generated export file.
    """
    try:
        export_job = db.query(ExportJob).filter(ExportJob.id == job_id).first()

        if not export_job:
            raise HTTPException(
                status_code=404,
                detail=f"Export job with ID {job_id} not found"
            )

        if export_job.status != ExportStatus.COMPLETED:
            raise HTTPException(
                status_code=400,
                detail=f"Export job {job_id} is not completed. Current status: {export_job.status.value}"
            )

        if not export_job.file_url:
            raise HTTPException(
                status_code=404,
                detail="Export file not available"
            )

        # Increment download count
        export_job.download_count += 1
        db.commit()

        # In a real implementation, you would serve the file from storage
        # For now, return a placeholder response
        file_content = f"Export file for job {job_id}: {export_job.title}".encode('utf-8')

        # Determine content type and filename
        if export_job.export_format.value == 'excel':
            content_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            filename = f"{export_job.title}.xlsx"
        elif export_job.export_format.value == 'word':
            content_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            filename = f"{export_job.title}.docx"
        elif export_job.export_format.value == 'pdf':
            content_type = 'application/pdf'
            filename = f"{export_job.title}.pdf"
        elif export_job.export_format.value == 'csv':
            content_type = 'text/csv'
            filename = f"{export_job.title}.csv"
        else:
            content_type = 'application/json'
            filename = f"{export_job.title}.json"

        return StreamingResponse(
            iter([file_content]),
            media_type=content_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading export file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error downloading export file: {str(e)}"
        )


@router.post("/preview", summary="Preview export configuration")
async def preview_export(
    export_data: ExportJobCreate,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Preview what the export would contain without actually generating the file.
    """
    try:
        # Query publications based on export criteria
        query = db.query(Publication)

        if export_data.faculty_ids:
            query = query.filter(Publication.faculty_id.in_(export_data.faculty_ids))

        if export_data.year_from:
            query = query.filter(Publication.publication_year >= export_data.year_from)

        if export_data.year_to:
            query = query.filter(Publication.publication_year <= export_data.year_to)

        if export_data.publication_types:
            query = query.filter(Publication.publication_type.in_(export_data.publication_types))

        publications = query.all()

        # Get faculty information
        faculty_ids = list(set(pub.faculty_id for pub in publications))
        faculty_list = db.query(Faculty).filter(Faculty.id.in_(faculty_ids)).all() if faculty_ids else []

        # Get plagiarism scans if requested
        plagiarism_scans = []
        if export_data.include_plagiarism_data:
            pub_ids = [pub.id for pub in publications]
            plagiarism_scans = db.query(PlagiarismScan).filter(PlagiarismScan.publication_id.in_(pub_ids)).all()

        # Generate preview statistics
        type_counts = {}
        year_counts = {}
        faculty_counts = {}

        for pub in publications:
            pub_type = pub.publication_type.value.replace('_', ' ').title()
            type_counts[pub_type] = type_counts.get(pub_type, 0) + 1

            year_counts[pub.publication_year] = year_counts.get(pub.publication_year, 0) + 1

            faculty_counts[pub.faculty_id] = faculty_counts.get(pub.faculty_id, 0) + 1

        preview = {
            "export_configuration": {
                "title": export_data.title,
                "export_type": export_data.export_type.value,
                "export_format": export_data.export_format.value,
                "include_plagiarism_data": export_data.include_plagiarism_data,
                "include_citations": export_data.include_citations,
                "include_impact_factors": export_data.include_impact_factors,
                "citation_style": export_data.citation_style
            },
            "content_preview": {
                "total_publications": len(publications),
                "total_faculty": len(faculty_ids),
                "total_plagiarism_scans": len(plagiarism_scans),
                "publication_types": type_counts,
                "years": dict(sorted(year_counts.items())),
                "faculty_count_by_id": faculty_counts
            },
            "estimated_file_size": {
                "excel": f"~{len(publications) * 2} KB",
                "word": f"~{len(publications) * 5} KB",
                "csv": f"~{len(publications) * 1} KB",
                "json": f"~{len(publications) * 3} KB"
            }
        }

        return {
            "success": True,
            "message": "Export preview generated successfully",
            "data": preview
        }

    except Exception as e:
        logger.error(f"Error generating export preview: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating export preview: {str(e)}"
        )


@router.get("/templates", summary="Get export templates")
async def get_export_templates() -> Dict[str, Any]:
    """
    Get predefined export templates for common use cases.
    """
    templates = {
        "accreditation_report": {
            "name": "Accreditation Report",
            "description": "Comprehensive report for accreditation agencies with year-wise organization",
            "export_type": "year_wise",
            "export_format": "excel",
            "include_citations": True,
            "include_impact_factors": True,
            "include_plagiarism_data": True,
            "default_parameters": {
                "include_statistics": True,
                "include_charts": True,
                "organization": "by_year_and_department"
            }
        },
        "faculty_portfolio": {
            "name": "Faculty Portfolio",
            "description": "Individual faculty publication portfolio for performance review",
            "export_type": "faculty_wise",
            "export_format": "word",
            "include_citations": True,
            "include_impact_factors": True,
            "include_plagiarism_data": False,
            "default_parameters": {
                "include_biography": True,
                "include_summary_statistics": True,
                "citation_style": "APA"
            }
        },
        "department_summary": {
            "name": "Department Summary",
            "description": "Department-wide publication summary with statistics",
            "export_type": "department_wise",
            "export_format": "excel",
            "include_citations": True,
            "include_impact_factors": True,
            "include_plagiarism_data": False,
            "default_parameters": {
                "include_collaboration_network": True,
                "include_productivity_metrics": True
            }
        },
        "plagiarism_audit": {
            "name": "Plagiarism Audit Report",
            "description": "Detailed plagiarism scan results and analysis",
            "export_type": "plagiarism_report",
            "export_format": "excel",
            "include_citations": False,
            "include_impact_factors": False,
            "include_plagiarism_data": True,
            "default_parameters": {
                "include_source_details": True,
                "include_risk_assessment": True,
                "recommendation_threshold": 0.3
            }
        },
        "research_productivity": {
            "name": "Research Productivity Dashboard",
            "description": "Analytics dashboard for research productivity metrics",
            "export_type": "summary_dashboard",
            "export_format": "excel",
            "include_citations": True,
            "include_impact_factors": True,
            "include_plagiarism_data": False,
            "default_parameters": {
                "include_trends": True,
                "include_benchmarks": True,
                "include_collaboration_metrics": True
            }
        }
    }

    return {
        "success": True,
        "data": {
            "templates": templates,
            "export_types": [
                {
                    "value": "faculty_wise",
                    "label": "Faculty Wise",
                    "description": "Organize publications by faculty member"
                },
                {
                    "value": "year_wise",
                    "label": "Year Wise",
                    "description": "Organize publications by publication year"
                },
                {
                    "value": "type_wise",
                    "label": "Type Wise",
                    "description": "Organize by publication type (journal, conference, etc.)"
                },
                {
                    "value": "department_wise",
                    "label": "Department Wise",
                    "description": "Group publications by department"
                },
                {
                    "value": "custom_duration",
                    "label": "Custom Duration",
                    "description": "Publications within specified date range"
                },
                {
                    "value": "summary_dashboard",
                    "label": "Summary Dashboard",
                    "description": "Overview statistics and summaries"
                },
                {
                    "value": "plagiarism_report",
                    "label": "Plagiarism Report",
                    "description": "Include plagiarism scan results"
                }
            ],
            "export_formats": [
                {"value": "excel", "label": "Excel (.xlsx)", "description": "Spreadsheet with multiple sheets"},
                {"value": "word", "label": "Word (.docx)", "description": "Formatted document"},
                {"value": "csv", "label": "CSV (.csv)", "description": "Comma-separated values"},
                {"value": "json", "label": "JSON (.json)", "description": "Structured data format"}
            ]
        }
    }


async def process_export_job(job_id: int, db: Session):
    """
    Background task to process export job.
    """
    try:
        # Get export job
        export_job = db.query(ExportJob).filter(ExportJob.id == job_id).first()
        if not export_job:
            logger.error(f"Export job {job_id} not found")
            return

        # Update status to in progress
        export_job.status = ExportStatus.IN_PROGRESS
        export_job.started_at = datetime.utcnow()
        db.commit()

        # Get publications based on criteria
        query = db.query(Publication)

        if export_job.faculty_ids:
            query = query.filter(Publication.faculty_id.in_(export_job.faculty_ids))

        if export_job.year_from:
            query = query.filter(Publication.publication_year >= export_job.year_from)

        if export_job.year_to:
            query = query.filter(Publication.publication_year <= export_job.year_to)

        if export_job.publication_types:
            query = query.filter(Publication.publication_type.in_(export_job.publication_types))

        publications = query.all()
        export_job.total_records = len(publications)
        db.commit()

        # Get faculty and plagiarism data if needed
        faculty_list = None
        if export_job.faculty_ids:
            faculty_list = db.query(Faculty).filter(Faculty.id.in_(export_job.faculty_ids)).all()

        plagiarism_scans = None
        if export_job.include_plagiarism_data and publications:
            pub_ids = [pub.id for pub in publications]
            plagiarism_scans = db.query(PlagiarismScan).filter(PlagiarismScan.publication_id.in_(pub_ids)).all()

        # Update progress
        export_job.processed_records = len(publications) // 2
        db.commit()

        # Generate export
        generator = ExportGenerator()
        file_content = await generator.generate_export(
            export_job=export_job,
            publications=publications,
            faculty=faculty_list,
            plagiarism_scans=plagiarism_scans
        )

        # Update job with completion details
        export_job.status = ExportStatus.COMPLETED
        export_job.completed_at = datetime.utcnow()
        export_job.processed_records = len(publications)
        export_job.progress_percentage = 100.0
        export_job.file_size_bytes = len(file_content)

        # In a real implementation, save file to cloud storage
        # For now, just set a placeholder URL
        export_job.file_url = f"/exports/{job_id}/{export_job.title.replace(' ', '_')}.{export_job.export_format.value}"

        db.commit()

        logger.info(f"Export job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"Error processing export job {job_id}: {str(e)}")

        # Update job with error
        if export_job:
            export_job.status = ExportStatus.FAILED
            export_job.error_message = str(e)
            export_job.completed_at = datetime.utcnow()
            db.commit()