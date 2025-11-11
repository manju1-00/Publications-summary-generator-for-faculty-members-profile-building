"""
Excel processor service for faculty data import and validation.
Handles Excel file parsing, validation, and faculty name normalization.
"""

import pandas as pd
import re
from typing import List, Dict, Tuple, Optional
from fastapi import HTTPException, UploadFile
from io import BytesIO
import unicodedata
from sqlalchemy.orm import Session

from app.models.faculty import Faculty, FacultyCreate, FacultyBatchUpload
from app.utils.validation import validate_email, validate_department, normalize_faculty_name


class ExcelProcessor:
    """Service for processing Excel files containing faculty information."""

    REQUIRED_COLUMNS = ['faculty_name', 'department', 'email']
    OPTIONAL_COLUMNS = ['title', 'specialization', 'research_interests']
    MAX_FILE_SIZE_MB = 50
    MAX_ROWS = 500
    SUPPORTED_FORMATS = ['.xlsx', '.xls']

    def __init__(self):
        self.departments_cache = {}
        self.errors = []
        self.warnings = []

    async def process_excel_file(
        self,
        file: UploadFile,
        db: Session
    ) -> FacultyBatchUpload:
        """
        Process uploaded Excel file and create faculty batch upload.

        Args:
            file: Uploaded Excel file
            db: Database session

        Returns:
            FacultyBatchUpload with processed faculty data

        Raises:
            HTTPException: If file processing fails
        """
        # Validate file
        await self._validate_file(file)

        # Read Excel file
        try:
            df = pd.read_excel(
                BytesIO(await file.read()),
                sheet_name=0,
                engine='openpyxl'
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to read Excel file: {str(e)}"
            )

        # Validate and clean data
        df = await self._validate_and_clean_dataframe(df)

        # Process faculty records
        faculty_records = await self._process_faculty_records(df)

        # Check for duplicates within the batch
        await self._check_batch_duplicates(faculty_records)

        # Check for duplicates in database
        await self._check_database_duplicates(faculty_records, db)

        # Create batch upload object
        batch_upload = FacultyBatchUpload(
            faculty_list=faculty_records,
            total_count=len(faculty_records)
        )

        return batch_upload

    async def _validate_file(self, file: UploadFile) -> None:
        """Validate uploaded Excel file."""
        # Check file extension
        file_extension = file.filename.lower().split('.')[-1]
        if f'.{file_extension}' not in self.SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        # Check file size
        if hasattr(file, 'size') and file.size > self.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {self.MAX_FILE_SIZE_MB}MB"
            )

        # Check content type
        allowed_content_types = [
            'application/vnd.ms-excel',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/octet-stream'
        ]
        if file.content_type not in allowed_content_types:
            self.warnings.append(
                f"Unexpected content type: {file.content_type}. Attempting to process anyway."
            )

    async def _validate_and_clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the DataFrame."""
        # Check if DataFrame is empty
        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="Excel file is empty or could not be read"
            )

        # Check maximum rows
        if len(df) > self.MAX_ROWS:
            raise HTTPException(
                status_code=400,
                detail=f"Too many rows. Maximum allowed: {self.MAX_ROWS}"
            )

        # Normalize column names (case-insensitive, strip whitespace)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

        # Check required columns
        missing_columns = []
        normalized_required = [col.lower().replace(' ', '_') for col in self.REQUIRED_COLUMNS]

        for req_col in normalized_required:
            if req_col not in df.columns:
                # Try to find similar column names
                similar_cols = [col for col in df.columns if req_col in col or col in req_col]
                if similar_cols:
                    df.rename(columns={similar_cols[0]: req_col}, inplace=True)
                    self.warnings(f"Using column '{similar_cols[0]}' for '{req_col}'")
                else:
                    missing_columns.append(req_col)

        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {', '.join(missing_columns)}. "
                       f"Required columns: {', '.join(self.REQUIRED_COLUMNS)}"
            )

        # Remove completely empty rows
        df = df.dropna(how='all')

        # Reset index
        df = df.reset_index(drop=True)

        return df

    async def _process_faculty_records(self, df: pd.DataFrame) -> List[FacultyCreate]:
        """Process DataFrame rows into faculty records."""
        faculty_records = []

        for index, row in df.iterrows():
            try:
                # Extract and validate faculty name
                faculty_name = self._extract_string_value(row, 'faculty_name')
                if not faculty_name:
                    self.errors.append(f"Row {index + 2}: Faculty name is required")
                    continue

                normalized_name = normalize_faculty_name(faculty_name)

                # Extract and validate department
                department = self._extract_string_value(row, 'department')
                if not department:
                    self.errors.append(f"Row {index + 2}: Department is required")
                    continue

                department = validate_department(department.strip())

                # Extract and validate email
                email = self._extract_string_value(row, 'email')
                if not email:
                    self.errors.append(f"Row {index + 2}: Email is required")
                    continue

                if not validate_email(email):
                    self.errors.append(f"Row {index + 2}: Invalid email format: {email}")
                    continue

                # Extract optional fields
                title = self._extract_string_value(row, 'title')
                specialization = self._extract_string_value(row, 'specialization')
                research_interests = self._extract_string_value(row, 'research_interests')

                # Create faculty record
                faculty_record = FacultyCreate(
                    name=faculty_name.strip(),
                    department=department,
                    email=email.lower().strip(),
                    title=title.strip() if title else None,
                    specialization=specialization.strip() if specialization else None,
                    research_interests=research_interests.strip() if research_interests else None
                )

                faculty_records.append(faculty_record)

            except Exception as e:
                self.errors.append(f"Row {index + 2}: {str(e)}")
                continue

        if self.errors:
            error_summary = f"Found {len(self.errors)} errors. First 5: {'; '.join(self.errors[:5])}"
            if len(self.errors) > 5:
                error_summary += f" (and {len(self.errors) - 5} more)"

            if len(faculty_records) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"No valid faculty records found. {error_summary}"
                )
            else:
                self.warnings.append(error_summary)

        return faculty_records

    def _extract_string_value(self, row: pd.Series, column: str) -> Optional[str]:
        """Extract string value from DataFrame row, handling various data types."""
        try:
            value = row.get(column)
            if value is None or pd.isna(value):
                return None

            # Convert to string and clean
            if isinstance(value, (int, float)):
                if value == int(value):  # It's an integer
                    value = str(int(value))
                else:
                    value = str(value)

            value = str(value).strip()

            # Handle empty strings
            if not value or value.lower() in ['nan', 'null', 'none', '']:
                return None

            return value
        except Exception:
            return None

    async def _check_batch_duplicates(self, faculty_records: List[FacultyCreate]) -> None:
        """Check for duplicate records within the batch."""
        email_counts = {}
        name_counts = {}

        for record in faculty_records:
            # Check email duplicates
            email_counts[record.email] = email_counts.get(record.email, 0) + 1
            # Check name duplicates (normalized)
            normalized_name = normalize_faculty_name(record.name)
            name_counts[normalized_name] = name_counts.get(normalized_name, 0) + 1

        # Report duplicates
        for email, count in email_counts.items():
            if count > 1:
                self.warnings.append(f"Duplicate email found in batch: {email} ({count} occurrences)")

        for name, count in name_counts.items():
            if count > 1:
                self.warnings.append(f"Similar faculty name found in batch: {name} ({count} occurrences)")

    async def _check_database_duplicates(
        self,
        faculty_records: List[FacultyCreate],
        db: Session
    ) -> None:
        """Check for duplicates in existing database."""
        emails_to_check = [record.email for record in faculty_records]

        if not emails_to_check:
            return

        # Query existing faculty with these emails
        existing_faculty = db.query(Faculty).filter(
            Faculty.email.in_(emails_to_check)
        ).all()

        existing_emails = {faculty.email for faculty in existing_faculty}

        # Report duplicates
        for record in faculty_records:
            if record.email in existing_emails:
                self.warnings.append(f"Faculty with email {record.email} already exists in database")

        if existing_faculty:
            self.warnings.append(
                f"Found {len(existing_faculty)} existing faculty members with same emails"
            )

    async def generate_preview(self, file: UploadFile) -> Dict:
        """Generate a preview of the Excel file without processing."""
        try:
            # Validate file
            await self._validate_file(file)

            # Read Excel file
            df = pd.read_excel(
                BytesIO(await file.read()),
                sheet_name=0,
                nrows=10,  # Only read first 10 rows for preview
                engine='openpyxl'
            )

            # Get file info
            file_info = {
                "filename": file.filename,
                "size_mb": round(len(await file.read()) / (1024 * 1024), 2),
                "total_rows": len(df),
                "columns": list(df.columns),
                "preview_data": df.head(5).to_dict('records'),
                "detected_columns": {
                    "required": [col for col in self.REQUIRED_COLUMNS
                               if any(req.lower() in col.lower() for req in df.columns)],
                    "optional": [col for col in self.OPTIONAL_COLUMNS
                               if any(opt.lower() in col.lower() for opt in df.columns)]
                }
            }

            return file_info

        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to generate preview: {str(e)}"
            )

    def get_errors_and_warnings(self) -> Dict[str, List[str]]:
        """Get accumulated errors and warnings."""
        return {
            "errors": self.errors,
            "warnings": self.warnings
        }


class FacultyNameNormalizer:
    """Utility class for normalizing faculty names for database queries."""

    @staticmethod
    def normalize_name(name: str) -> str:
        """Normalize faculty name for consistent matching."""
        if not name:
            return ""

        # Remove accents and normalize unicode
        name = unicodedata.normalize('NFKD', name)
        name = ''.join([c for c in name if not unicodedata.combining(c)])

        # Convert to lowercase and remove extra whitespace
        name = name.lower().strip()

        # Remove punctuation except hyphens and apostrophes
        name = re.sub(r'[^\w\s\-\'\.]', '', name)

        # Replace multiple spaces with single space
        name = re.sub(r'\s+', ' ', name)

        return name.strip()

    @staticmethod
    def generate_name_variations(name: str) -> List[str]:
        """Generate common variations of faculty names for database queries."""
        if not name:
            return []

        variations = []

        # Original and normalized
        variations.append(name)
        normalized = FacultyNameNormalizer.normalize_name(name)
        variations.append(normalized)

        # Split name into parts
        name_parts = name.split()
        if len(name_parts) >= 2:
            # First initial + last name
            first_initial = name_parts[0][0].upper() + '.'
            last_name = ' '.join(name_parts[1:])
            variations.append(f"{first_initial} {last_name}")

            # First name + last name (normalized)
            if len(name_parts) == 2:
                variations.append(f"{name_parts[1]}, {name_parts[0]}")

            # All initials
            initials = ''.join([part[0].upper() for part in name_parts if part])
            variations.append(initials)

            # Middle initial variations
            if len(name_parts) >= 3:
                middle_initial = name_parts[1][0].upper() + '.'
                variations.append(f"{name_parts[0]} {middle_initial} {' '.join(name_parts[2:])}")
                variations.append(f"{first_initial} {middle_initial} {' '.join(name_parts[2:])}")

        # Remove duplicates while preserving order
        unique_variations = []
        seen = set()
        for var in variations:
            normalized_var = FacultyNameNormalizer.normalize_name(var)
            if normalized_var not in seen:
                seen.add(normalized_var)
                unique_variations.append(var)

        return unique_variations