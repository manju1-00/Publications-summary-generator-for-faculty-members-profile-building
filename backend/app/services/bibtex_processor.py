"""
BibTeX processor service for handling academic publication imports.
Parses BibTeX files and extracts faculty names and publication data.
"""

import bibtexparser
from bibtexparser.bibdatabase import BibDatabase
from bibtexparser.customization import convert_to_unicode
from typing import List, Dict, Set, Optional, Tuple
from fastapi import HTTPException, UploadFile
from io import BytesIO
import re
from datetime import datetime
import unicodedata

from app.models.publication import Publication, PublicationCreate, PublicationType
from app.utils.validation import validate_publication_title, validate_publication_year, validate_doi
from app.utils.validation import normalize_faculty_name, generate_name_variations


class BibTeXProcessor:
    """Service for processing BibTeX files and extracting publication data."""

    SUPPORTED_ENTRY_TYPES = {
        'article': PublicationType.JOURNAL,
        'inproceedings': PublicationType.CONFERENCE,
        'inbook': PublicationType.BOOK_CHAPTER,
        'book': PublicationType.BOOK,
        'phdthesis': PublicationType.THESIS,
        'mastersthesis': PublicationType.THESIS,
        'techreport': PublicationType.TECHNICAL_REPORT,
        'misc': PublicationType.OTHER,
        'preprint': PublicationType.PREPRINT,
        'workshop': PublicationType.WORKSHOP,
    }

    REQUIRED_FIELDS = ['title', 'author', 'year']
    OPTIONAL_FIELDS = [
        'journal', 'booktitle', 'publisher', 'volume', 'issue', 'pages',
        'doi', 'isbn', 'issn', 'url', 'abstract', 'month', 'note'
    ]

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.processed_entries = 0
        self.faculty_names_found = set()

    async def process_bibtex_file(self, file: UploadFile) -> Dict:
        """
        Process uploaded BibTeX file and extract publication data.

        Args:
            file: Uploaded BibTeX file

        Returns:
            Dictionary containing processed data and metadata

        Raises:
            HTTPException: If file processing fails
        """
        # Validate file
        await self._validate_file(file)

        # Read and parse BibTeX file
        try:
            content = await file.read()
            bibtex_str = content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                # Try different encodings
                content = await file.read()
                bibtex_str = content.decode('latin-1')
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to decode BibTeX file: {str(e)}"
                )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to read BibTeX file: {str(e)}"
            )

        # Parse BibTeX content
        try:
            parser = bibtexparser.bparser.BibTexParser(
                common_strings=True,
                ignore_nonstandard_types=False,
                homogenize_fields=True,
                customize=convert_to_unicode
            )
            bib_database = bibtexparser.loads(bibtex_str, parser=parser)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to parse BibTeX file: {str(e)}"
            )

        # Process entries
        publications = []
        faculty_names = set()

        for entry in bib_database.entries:
            try:
                publication_data = await self._process_entry(entry)
                if publication_data:
                    publications.append(publication_data)

                # Extract faculty names from author field
                entry_faculty_names = self._extract_author_names(entry.get('author', ''))
                faculty_names.update(entry_faculty_names)

                self.processed_entries += 1

            except Exception as e:
                self.errors.append(f"Entry {entry.get('ID', 'Unknown')}: {str(e)}")
                continue

        # Generate report
        result = {
            'publications': publications,
            'faculty_names': list(faculty_names),
            'total_entries': len(bib_database.entries),
            'processed_entries': self.processed_entries,
            'publication_types_count': self._count_publication_types(publications),
            'years_covered': self._get_years_covered(publications),
            'errors': self.errors,
            'warnings': self.warnings
        }

        return result

    async def _validate_file(self, file: UploadFile) -> None:
        """Validate uploaded BibTeX file."""
        # Check file extension
        file_extension = file.filename.lower().split('.')[-1]
        if file_extension != 'bib':
            raise HTTPException(
                status_code=400,
                detail="Invalid file extension. Expected .bib file"
            )

        # Check file size (max 10MB for BibTeX files)
        max_size = 10 * 1024 * 1024  # 10MB
        if hasattr(file, 'size') and file.size > max_size:
            raise HTTPException(
                status_code=400,
                detail="File too large. Maximum size: 10MB"
            )

        # Check content type
        allowed_content_types = [
            'application/x-bibtex',
            'text/plain',
            'text/x-bibtex',
            'application/octet-stream'
        ]
        if file.content_type and file.content_type not in allowed_content_types:
            self.warnings.append(
                f"Unexpected content type: {file.content_type}. Attempting to process anyway."
            )

    async def _process_entry(self, entry: Dict) -> Optional[PublicationCreate]:
        """Process individual BibTeX entry and create publication record."""
        # Check required fields
        missing_fields = []
        for field in self.REQUIRED_FIELDS:
            if field not in entry or not entry[field].strip():
                missing_fields.append(field)

        if missing_fields:
            self.errors.append(
                f"Entry {entry.get('ID', 'Unknown')}: Missing required fields: {', '.join(missing_fields)}"
            )
            return None

        try:
            # Extract and validate title
            title = validate_publication_title(entry['title'])

            # Extract and validate year
            try:
                year = int(entry['year'])
                year = validate_publication_year(year)
            except ValueError:
                self.errors.append(
                    f"Entry {entry.get('ID', 'Unknown')}: Invalid year: {entry['year']}"
                )
                return None

            # Determine publication type
            entry_type = entry.get('type', 'misc').lower()
            publication_type = self.SUPPORTED_ENTRY_TYPES.get(entry_type, PublicationType.OTHER)

            # Extract authors
            authors = self._parse_authors(entry['author'])

            # Extract venue information
            venue_name = None
            if publication_type == PublicationType.JOURNAL:
                venue_name = entry.get('journal')
            elif publication_type == PublicationType.CONFERENCE:
                venue_name = entry.get('booktitle')
            elif publication_type == PublicationType.BOOK:
                venue_name = entry.get('publisher')

            # Extract publication metadata
            doi = validate_doi(entry.get('doi'))
            isbn = entry.get('isbn')
            issn = entry.get('issn')
            url = entry.get('url')
            abstract = entry.get('abstract')

            # Extract additional metadata
            volume = entry.get('volume')
            issue = entry.get('issue')
            pages = entry.get('pages')
            publisher = entry.get('publisher')

            # Create publication record
            publication = PublicationCreate(
                faculty_id=1,  # Will be updated after faculty matching
                title=title,
                abstract=abstract,
                publication_type=publication_type,
                venue_name=venue_name,
                publisher=publisher,
                volume=volume,
                issue=issue,
                pages=pages,
                doi=doi,
                isbn=isbn,
                issn=issn,
                url=url,
                publication_year=year,
                authors=authors,
                author_count=len(authors),
                source_database="bibtex_import",
                source_metadata={
                    'entry_id': entry.get('ID'),
                    'entry_type': entry.get('type'),
                    'raw_entry': entry
                }
            )

            return publication

        except Exception as e:
            self.errors.append(f"Entry {entry.get('ID', 'Unknown')}: {str(e)}")
            return None

    def _parse_authors(self, author_string: str) -> List[str]:
        """Parse author string into list of author names."""
        if not author_string:
            return []

        # Split by 'and' (BibTeX author separator)
        authors = re.split(r'\s+and\s+', author_string.strip(), flags=re.IGNORECASE)

        # Clean and normalize author names
        cleaned_authors = []
        for author in authors:
            author = author.strip()
            if author:
                # Remove braces
                author = re.sub(r'[{}]', '', author)

                # Normalize whitespace
                author = re.sub(r'\s+', ' ', author)

                # Handle special cases (e.g., "{von Neumann, John}")
                if ',' in author:
                    parts = author.split(',', 1)
                    if len(parts) == 2:
                        author = f"{parts[1].strip()} {parts[0].strip()}"

                cleaned_authors.append(author)

        return cleaned_authors

    def _extract_author_names(self, author_string: str) -> Set[str]:
        """Extract faculty names from author string."""
        if not author_string:
            return set()

        authors = self._parse_authors(author_string)
        faculty_names = set()

        for author in authors:
            # Generate variations for matching
            variations = generate_name_variations(author)
            faculty_names.update(variations)

        return faculty_names

    def _count_publication_types(self, publications: List[Dict]) -> Dict[str, int]:
        """Count publications by type."""
        type_counts = {}
        for pub in publications:
            pub_type = pub.publication_type.value if hasattr(pub.publication_type, 'value') else str(pub.publication_type)
            type_counts[pub_type] = type_counts.get(pub_type, 0) + 1
        return type_counts

    def _get_years_covered(self, publications: List[Dict]) -> List[int]:
        """Get list of years covered by publications."""
        years = set()
        for pub in publications:
            if hasattr(pub, 'publication_year'):
                years.add(pub.publication_year)
        return sorted(list(years))

    async def match_faculty_to_publications(
        self,
        publications: List[PublicationCreate],
        faculty_names: List[str]
    ) -> List[PublicationCreate]:
        """
        Match faculty names to publications and assign faculty IDs.

        Args:
            publications: List of publications to match
            faculty_names: List of known faculty names

        Returns:
            Updated publications with faculty IDs
        """
        # Create normalized faculty name variations for matching
        faculty_variations = {}
        for name in faculty_names:
            variations = generate_name_variations(name)
            for variation in variations:
                normalized = normalize_faculty_name(variation)
                faculty_variations[normalized] = name

        matched_publications = []

        for pub in publications:
            best_match = None
            best_score = 0

            if pub.authors:
                for author in pub.authors:
                    author_normalized = normalize_faculty_name(author)

                    # Check for exact match
                    if author_normalized in faculty_variations:
                        match_score = 1.0
                        if match_score > best_score:
                            best_match = faculty_variations[author_normalized]
                            best_score = match_score

                    # Check for partial match
                    for faculty_norm, faculty_name in faculty_variations.items():
                        if faculty_norm in author_normalized or author_normalized in faculty_norm:
                            match_score = 0.7
                            if match_score > best_score:
                                best_match = faculty_name
                                best_score = match_score

            if best_match:
                # Note: In a real implementation, you would query the database
                # to get the faculty_id for the matched name
                # For now, we'll store the matched name in source_metadata
                if pub.source_metadata is None:
                    pub.source_metadata = {}
                pub.source_metadata['matched_faculty'] = best_match
                pub.source_metadata['match_score'] = best_score

                self.warnings.append(
                    f"Publication '{pub.title[:50]}...' matched to faculty: {best_match} (score: {best_score})"
                )
            else:
                if pub.source_metadata is None:
                    pub.source_metadata = {}
                pub.source_metadata['matched_faculty'] = None
                pub.source_metadata['match_score'] = 0.0

                self.warnings.append(
                    f"No faculty match found for publication: '{pub.title[:50]}...'"
                )

            matched_publications.append(pub)

        return matched_publications

    async def generate_import_report(self, file: UploadFile) -> Dict:
        """Generate a preview report of BibTeX file without full processing."""
        try:
            # Validate file
            await self._validate_file(file)

            # Read file content
            content = await file.read()
            bibtex_str = content.decode('utf-8')

            # Quick parse to get entry count and basic info
            entries = re.findall(r'@[a-zA-Z]+\s*\{[^@]*\}', bibtex_str, re.DOTALL)

            entry_types = {}
            years = []
            authors_set = set()

            for entry in entries:
                # Extract entry type
                type_match = re.search(r'@([a-zA-Z]+)', entry)
                if type_match:
                    entry_type = type_match.group(1).lower()
                    entry_types[entry_type] = entry_types.get(entry_type, 0) + 1

                # Extract year
                year_match = re.search(r'year\s*=\s*{?(\d{4})}?', entry)
                if year_match:
                    years.append(int(year_match.group(1)))

                # Extract authors (basic extraction)
                author_match = re.search(r'author\s*=\s*{([^}]*)}', entry)
                if author_match:
                    authors = self._parse_authors(author_match.group(1))
                    authors_set.update(authors)

            report = {
                "filename": file.filename,
                "total_entries": len(entries),
                "entry_types": entry_types,
                "years_range": {
                    "min": min(years) if years else None,
                    "max": max(years) if years else None,
                    "count": len(set(years))
                },
                "unique_authors": len(authors_set),
                "sample_authors": list(authors_set)[:10],
                "file_size_kb": round(len(bibtex_str) / 1024, 2),
                "estimated_processing_time_minutes": round(len(entries) * 0.1, 1)  # Rough estimate
            }

            return report

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


class BibTeXValidator:
    """Utility class for validating BibTeX entries."""

    @staticmethod
    def validate_entry(entry: Dict) -> Tuple[bool, List[str]]:
        """
        Validate a single BibTeX entry.

        Args:
            entry: BibTeX entry dictionary

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check for entry ID
        if 'ID' not in entry or not entry['ID']:
            errors.append("Entry ID is missing")

        # Check required fields
        required_fields = ['title', 'author', 'year']
        for field in required_fields:
            if field not in entry or not entry[field].strip():
                errors.append(f"Required field '{field}' is missing or empty")

        # Validate year
        if 'year' in entry:
            try:
                year = int(entry['year'])
                if year < 1900 or year > datetime.now().year + 2:
                    errors.append(f"Invalid year: {year}")
            except ValueError:
                errors.append(f"Year is not a valid number: {entry['year']}")

        # Validate DOI format if present
        if 'doi' in entry and entry['doi']:
            doi = validate_doi(entry['doi'])
            if doi is None:
                errors.append(f"Invalid DOI format: {entry['doi']}")

        return len(errors) == 0, errors