"""
Validation utilities for faculty data, publications, and input validation.
"""

import re
from typing import Optional, List
from pydantic import EmailStr
import unicodedata


def validate_email(email: str) -> bool:
    """
    Validate email address format.

    Args:
        email: Email address to validate

    Returns:
        True if valid, False otherwise
    """
    if not email:
        return False

    # Basic email regex pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email.strip()) is not None


def validate_department(department: str, allowed_departments: Optional[List[str]] = None) -> str:
    """
    Validate and normalize department name.

    Args:
        department: Department name to validate
        allowed_departments: List of allowed department names

    Returns:
        Normalized department name

    Raises:
        ValueError: If department is invalid
    """
    if not department:
        raise ValueError("Department name is required")

    department = department.strip().title()

    # Standardize common department name variations
    department_mappings = {
        'cs': 'Computer Science',
        'cse': 'Computer Science and Engineering',
        'it': 'Information Technology',
        'ee': 'Electrical Engineering',
        'ece': 'Electrical and Computer Engineering',
        'me': 'Mechanical Engineering',
        'ce': 'Civil Engineering',
        'chem': 'Chemistry',
        'physics': 'Physics',
        'math': 'Mathematics',
        'bio': 'Biology',
        'biotech': 'Biotechnology',
        'mse': 'Materials Science and Engineering',
        'ise': 'Industrial and Systems Engineering',
        'arch': 'Architecture',
        'business': 'Business Administration',
        'econ': 'Economics',
        'psych': 'Psychology',
        'sociology': 'Sociology',
        'english': 'English Literature',
        'history': 'History',
        'philosophy': 'Philosophy',
    }

    # Handle common abbreviations
    lower_dept = department.lower()
    if lower_dept in department_mappings:
        department = department_mappings[lower_dept]

    # Check against allowed departments if provided
    if allowed_departments:
        allowed_lower = [dept.lower() for dept in allowed_departments]
        if department.lower() not in allowed_lower:
            raise ValueError(
                f"Department '{department}' is not in the allowed list. "
                f"Allowed departments: {', '.join(allowed_departments)}"
            )

    return department


def normalize_faculty_name(name: str) -> str:
    """
    Normalize faculty name for consistent database queries and matching.

    Args:
        name: Faculty name to normalize

    Returns:
        Normalized name string
    """
    if not name:
        return ""

    # Remove accents and normalize unicode
    name = unicodedata.normalize('NFKD', name)
    name = ''.join([c for c in name if not unicodedata.combining(c)])

    # Convert to lowercase
    name = name.lower()

    # Remove extra punctuation and normalize whitespace
    # Keep hyphens, apostrophes, and periods
    name = re.sub(r'[^\w\s\-\'\.]', '', name)
    name = re.sub(r'\s+', ' ', name)

    return name.strip()


def validate_publication_title(title: str) -> str:
    """
    Validate and normalize publication title.

    Args:
        title: Publication title to validate

    Returns:
        Normalized title

    Raises:
        ValueError: If title is invalid
    """
    if not title:
        raise ValueError("Publication title is required")

    title = title.strip()

    if len(title) < 5:
        raise ValueError("Publication title must be at least 5 characters long")

    if len(title) > 1000:
        raise ValueError("Publication title cannot exceed 1000 characters")

    # Remove excessive whitespace
    title = re.sub(r'\s+', ' ', title)

    return title


def validate_publication_year(year: int) -> int:
    """
    Validate publication year.

    Args:
        year: Year to validate

    Returns:
        Validated year

    Raises:
        ValueError: If year is invalid
    """
    from datetime import datetime

    current_year = datetime.now().year

    if year < 1900:
        raise ValueError("Publication year cannot be before 1900")

    if year > current_year + 2:  # Allow future publications up to 2 years
        raise ValueError(f"Publication year cannot be more than {current_year + 2}")

    return year


def validate_doi(doi: str) -> Optional[str]:
    """
    Validate DOI (Digital Object Identifier) format.

    Args:
        doi: DOI string to validate

    Returns:
        Normalized DOI or None if invalid

    Raises:
        ValueError: If DOI format is invalid
    """
    if not doi:
        return None

    doi = doi.strip()

    # Remove DOI prefix if present
    if doi.lower().startswith('doi:'):
        doi = doi[4:]

    # Basic DOI validation
    if not (doi.startswith('10.') and '/' in doi):
        return None  # Invalid format, but don't raise error

    # Clean DOI
    doi = re.sub(r'\s+', '', doi)

    return doi


def validate_isbn(isbn: str) -> Optional[str]:
    """
    Validate ISBN (International Standard Book Number) format.

    Args:
        isbn: ISBN string to validate

    Returns:
        Normalized ISBN or None if invalid
    """
    if not isbn:
        return None

    # Remove hyphens and spaces
    isbn = re.sub(r'[\s\-]', '', isbn)

    # Check for valid ISBN-10 or ISBN-13
    if len(isbn) == 10:
        # ISBN-10 validation
        if not re.match(r'^\d{9}[\dX]$', isbn):
            return None
    elif len(isbn) == 13:
        # ISBN-13 validation
        if not re.match(r'^\d{13}$', isbn):
            return None
    else:
        return None

    return isbn


def validate_issn(issn: str) -> Optional[str]:
    """
    Validate ISSN (International Standard Serial Number) format.

    Args:
        issn: ISSN string to validate

    Returns:
        Normalized ISSN or None if invalid
    """
    if not issn:
        return None

    # Remove hyphens and spaces
    issn = re.sub(r'[\s\-]', '', issn)

    # ISSN validation (8 digits, last can be X)
    if not re.match(r'^\d{7}[\dX]$', issn):
        return None

    return issn


def validate_url(url: str) -> Optional[str]:
    """
    Validate URL format.

    Args:
        url: URL string to validate

    Returns:
        Normalized URL or None if invalid
    """
    if not url:
        return None

    url = url.strip()

    # Basic URL validation
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    if not url_pattern.match(url):
        return None

    return url


def sanitize_text(text: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize text input by removing potentially harmful content.

    Args:
        text: Text to sanitize
        max_length: Maximum allowed length

    Returns:
        Sanitized text
    """
    if not text:
        return ""

    text = text.strip()

    # Remove potentially dangerous HTML/JS content
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<[^>]+>', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Apply length limit if specified
    if max_length and len(text) > max_length:
        text = text[:max_length].rsplit(' ', 1)[0]  # Truncate at last complete word

    return text


def validate_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
    """
    Validate file extension against allowed list.

    Args:
        filename: File name to validate
        allowed_extensions: List of allowed extensions (including dot)

    Returns:
        True if extension is allowed, False otherwise
    """
    if not filename:
        return False

    # Extract extension
    extension = '.' + filename.lower().split('.')[-1] if '.' in filename else ''

    return extension in [ext.lower() for ext in allowed_extensions]


def validate_file_size(file_size_bytes: int, max_size_mb: int) -> bool:
    """
    Validate file size against maximum allowed size.

    Args:
        file_size_bytes: File size in bytes
        max_size_mb: Maximum allowed size in megabytes

    Returns:
        True if size is within limit, False otherwise
    """
    max_size_bytes = max_size_mb * 1024 * 1024
    return file_size_bytes <= max_size_bytes


def validate_faculty_name(name: str) -> str:
    """
    Validate and normalize faculty name.

    Args:
        name: Faculty name to validate

    Returns:
        Normalized faculty name

    Raises:
        ValueError: If name is invalid
    """
    if not name:
        raise ValueError("Faculty name is required")

    name = name.strip()

    if len(name) < 2:
        raise ValueError("Faculty name must be at least 2 characters long")

    if len(name) > 255:
        raise ValueError("Faculty name cannot exceed 255 characters")

    # Check for valid characters (letters, spaces, hyphens, apostrophes, periods)
    if not re.match(r'^[a-zA-Z\s\-\'\.]+$', name):
        raise ValueError("Faculty name contains invalid characters")

    # Normalize whitespace
    name = re.sub(r'\s+', ' ', name)

    # Title case the name
    name = name.title()

    return name


def generate_name_variations(name: str) -> List[str]:
    """
    Generate common variations of faculty names for database queries.

    Args:
        name: Faculty name to generate variations for

    Returns:
        List of name variations
    """
    variations = []
    name_parts = name.split()

    if len(name_parts) >= 2:
        first_name = name_parts[0]
        last_name = ' '.join(name_parts[1:])

        # Original name
        variations.append(name)

        # First initial + last name
        first_initial = first_name[0] + '.'
        variations.append(f"{first_initial} {last_name}")

        # Last name + first name
        variations.append(f"{last_name}, {first_name}")

        # All initials
        initials = ''.join([part[0].upper() for part in name_parts])
        variations.append(initials)

    else:
        variations.append(name)

    # Remove duplicates while preserving order
    unique_variations = []
    seen = set()
    for var in variations:
        normalized = normalize_faculty_name(var)
        if normalized not in seen:
            seen.add(normalized)
            unique_variations.append(var)

    return unique_variations