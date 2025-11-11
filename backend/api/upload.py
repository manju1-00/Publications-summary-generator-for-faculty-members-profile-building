"""
Upload API endpoints for faculty data import.
Handles Excel and BibTeX file uploads with validation and processing.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
import logging

from app.utils.database import get_db
from app.services.excel_processor import ExcelProcessor
from app.services.bibtex_processor import BibTeXProcessor
from app.models.faculty import Faculty

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/excel", summary="Upload Excel file with faculty data")
async def upload_excel_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Upload Excel file containing faculty information.

    The Excel file should contain the following columns:
    - Faculty Name: Full name of faculty member
    - Department: Department name
    - Email: Faculty email address
    - Title (optional): Faculty title (Professor, Associate Professor, etc.)
    - Specialization (optional): Area of specialization
    - Research Interests (optional): Research interests

    Maximum file size: 50MB
    Maximum rows: 500 faculty members
    """
    try:
        # Initialize Excel processor
        processor = ExcelProcessor()

        # Process the Excel file
        batch_upload = await processor.process_excel_file(file, db)

        # Get errors and warnings
        errors_warnings = processor.get_errors_and_warnings()

        response = {
            "success": True,
            "message": f"Successfully processed {batch_upload.total_count} faculty records",
            "data": {
                "total_count": batch_upload.total_count,
                "faculty_list": [
                    {
                        "name": fac.name,
                        "department": fac.department,
                        "email": fac.email,
                        "title": fac.title,
                        "specialization": fac.specialization
                    }
                    for fac in batch_upload.faculty_list
                ]
            },
            "errors": errors_warnings["errors"],
            "warnings": errors_warnings["warnings"]
        }

        # Add warnings to response if present
        if errors_warnings["warnings"]:
            response["warning_message"] = f"Completed with {len(errors_warnings['warnings'])} warnings"

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing Excel file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing Excel file: {str(e)}"
        )


@router.post("/excel/preview", summary="Preview Excel file before upload")
async def preview_excel_file(
    file: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Generate a preview of Excel file without processing all records.
    Useful for validating file format before full upload.
    """
    try:
        # Initialize Excel processor
        processor = ExcelProcessor()

        # Generate preview
        preview = await processor.generate_preview(file)

        return {
            "success": True,
            "message": "Excel file preview generated successfully",
            "data": preview
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating Excel preview: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating Excel preview: {str(e)}"
        )


@router.post("/bibtex", summary="Upload BibTeX file with publication data")
async def upload_bibtex_file(
    file: UploadFile = File(...),
    process_faculty: bool = False,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Upload BibTeX file containing publication data.

    This endpoint processes BibTeX files and extracts:
    - Publication metadata (title, authors, year, venue, etc.)
    - Faculty names from author fields
    - Publication categorization (journal vs conference)

    Parameters:
    - process_faculty: If True, attempts to match authors to existing faculty
    """
    try:
        # Initialize BibTeX processor
        processor = BibTeXProcessor()

        # Process the BibTeX file
        result = await processor.process_bibtex_file(file)

        # Optionally match to faculty
        if process_faculty:
            # Get existing faculty from database
            faculty_list = db.query(Faculty).all()
            faculty_names = [fac.name for fac in faculty_list]
            faculty_mapping = {fac.name: fac.id for fac in faculty_list}

            # Match publications to faculty
            matched_publications = await processor.match_faculty_to_publications(
                result['publications'], faculty_names
            )
            result['publications'] = matched_publications
            result['faculty_matched'] = True
        else:
            result['faculty_matched'] = False

        # Get errors and warnings
        errors_warnings = processor.get_errors_and_warnings()

        response = {
            "success": True,
            "message": f"Successfully processed {result['processed_entries']} BibTeX entries",
            "data": {
                "total_entries": result["total_entries"],
                "processed_entries": result["processed_entries"],
                "publication_count": len(result["publications"]),
                "faculty_names_found": result["faculty_names"],
                "publication_types_count": result["publication_types_count"],
                "years_covered": result["years_covered"],
                "faculty_matched": result.get("faculty_matched", False),
                "publications": [
                    {
                        "title": pub.title,
                        "authors": pub.authors,
                        "year": pub.publication_year,
                        "type": pub.publication_type.value,
                        "venue": pub.venue_name,
                        "doi": pub.doi
                    }
                    for pub in result["publications"][:10]  # First 10 for preview
                ]
            },
            "errors": errors_warnings["errors"],
            "warnings": errors_warnings["warnings"]
        }

        # Add warnings if present
        if errors_warnings["warnings"]:
            response["warning_message"] = f"Completed with {len(errors_warnings['warnings'])} warnings"

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing BibTeX file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing BibTeX file: {str(e)}"
        )


@router.post("/bibtex/preview", summary="Preview BibTeX file before upload")
async def preview_bibtex_file(
    file: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Generate a preview of BibTeX file without processing all records.
    """
    try:
        # Initialize BibTeX processor
        processor = BibTeXProcessor()

        # Generate preview
        preview = await processor.generate_import_report(file)

        return {
            "success": True,
            "message": "BibTeX file preview generated successfully",
            "data": preview
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating BibTeX preview: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating BibTeX preview: {str(e)}"
        )


@router.get("/formats", summary="Get supported file formats")
async def get_supported_formats() -> Dict[str, Any]:
    """
    Get information about supported file formats and their requirements.
    """
    excel_formats = {
        "extensions": [".xlsx", ".xls"],
        "mime_types": [
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/octet-stream"
        ],
        "max_size_mb": 50,
        "max_rows": 500,
        "required_columns": [
            {
                "name": "Faculty Name",
                "description": "Full name of the faculty member",
                "required": True
            },
            {
                "name": "Department",
                "description": "Department name",
                "required": True
            },
            {
                "name": "Email",
                "description": "Faculty email address",
                "required": True
            }
        ],
        "optional_columns": [
            {
                "name": "Title",
                "description": "Faculty title (Professor, Associate Professor, etc.)",
                "required": False
            },
            {
                "name": "Specialization",
                "description": "Area of specialization",
                "required": False
            },
            {
                "name": "Research Interests",
                "description": "Research interests",
                "required": False
            }
        ]
    }

    bibtex_formats = {
        "extensions": [".bib"],
        "mime_types": [
            "application/x-bibtex",
            "text/plain",
            "text/x-bibtex",
            "application/octet-stream"
        ],
        "max_size_mb": 10,
        "supported_entry_types": [
            "article", "inproceedings", "inbook", "book",
            "phdthesis", "mastersthesis", "techreport", "misc",
            "preprint", "workshop"
        ],
        "required_fields": ["title", "author", "year"],
        "optional_fields": [
            "journal", "booktitle", "publisher", "volume", "issue",
            "pages", "doi", "isbn", "issn", "url", "abstract", "month", "note"
        ]
    }

    return {
        "success": True,
        "data": {
            "excel": excel_formats,
            "bibtex": bibtex_formats,
            "general_guidelines": [
                "Files are processed asynchronously for large datasets",
                "Excel files should have column names in the first row",
                "BibTeX files must be properly formatted according to standard specifications",
                "Email addresses are validated for correct format",
                "Faculty names are normalized for consistent database queries"
            ]
        }
    }


@router.post("/validate", summary="Validate uploaded file without processing")
async def validate_uploaded_file(
    file: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Validate uploaded file format and structure without processing data.
    """
    try:
        file_extension = file.filename.lower().split('.')[-1]

        if file_extension in ['xlsx', 'xls']:
            # Validate Excel file
            processor = ExcelProcessor()
            await processor._validate_file(file)
            preview = await processor.generate_preview(file)

            return {
                "success": True,
                "message": "Excel file is valid",
                "data": {
                    "file_type": "excel",
                    "validation_status": "valid",
                    "preview": preview
                }
            }

        elif file_extension == 'bib':
            # Validate BibTeX file
            processor = BibTeXProcessor()
            await processor._validate_file(file)
            preview = await processor.generate_import_report(file)

            return {
                "success": True,
                "message": "BibTeX file is valid",
                "data": {
                    "file_type": "bibtex",
                    "validation_status": "valid",
                    "preview": preview
                }
            }

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: .{file_extension}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error validating file: {str(e)}"
        )


@router.get("/status", summary="Get upload service status")
async def get_upload_status() -> Dict[str, Any]:
    """
    Get the current status of the upload service.
    """
    return {
        "success": True,
        "data": {
            "service_status": "active",
            "supported_formats": ["excel", "bibtex"],
            "max_file_size": {
                "excel": "50MB",
                "bibtex": "10MB"
            },
            "processing_limits": {
                "max_faculty_per_excel": 500,
                "max_entries_per_bibtex": 10000
            },
            "features": [
                "Excel file validation and processing",
                "BibTeX file parsing and extraction",
                "Faculty name normalization and matching",
                "Publication type detection",
                "Duplicate detection",
                "Error reporting with detailed messages"
            ]
        }
    }