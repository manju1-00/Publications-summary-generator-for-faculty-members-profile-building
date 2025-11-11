"""
Publication processing service for deduplication and categorization.
Handles publication type detection, deduplication, and quality scoring.
"""

import re
from typing import List, Dict, Set, Tuple, Optional
from datetime import datetime
from difflib import SequenceMatcher
import numpy as np
from collections import defaultdict

from app.models.publication import PublicationCreate, PublicationType
from app.utils.validation import normalize_faculty_name, validate_doi


class PublicationProcessor:
    """Service for processing and categorizing publications."""

    def __init__(self):
        self.venue_patterns = self._load_venue_patterns()
        self.conference_keywords = self._load_conference_keywords()
        self.journal_keywords = self._load_journal_keywords()
        self.publisher_patterns = self._load_publisher_patterns()

    async def process_publications(
        self,
        publications: List[PublicationCreate],
        faculty_mapping: Optional[Dict[str, int]] = None
    ) -> Tuple[List[PublicationCreate], Dict]:
        """
        Process publications: deduplicate, categorize, and match to faculty.

        Args:
            publications: List of publications to process
            faculty_mapping: Mapping of faculty names to faculty IDs

        Returns:
            Tuple of (processed_publications, processing_statistics)
        """
        if not publications:
            return [], {}

        # Step 1: Clean and normalize publications
        cleaned_publications = await self._clean_publications(publications)

        # Step 2: Deduplicate publications
        deduplicated_publications = await self._deduplicate_publications(cleaned_publications)

        # Step 3: Categorize publications
        categorized_publications = await self._categorize_publications(deduplicated_publications)

        # Step 4: Match publications to faculty
        if faculty_mapping:
            categorized_publications = await self._match_to_faculty(
                categorized_publications, faculty_mapping
            )

        # Step 5: Calculate quality scores
        scored_publications = await self._calculate_quality_scores(categorized_publications)

        # Step 6: Generate statistics
        stats = await self._generate_processing_statistics(publications, scored_publications)

        return scored_publications, stats

    async def _clean_publications(self, publications: List[PublicationCreate]) -> List[PublicationCreate]:
        """Clean and normalize publication data."""
        cleaned = []

        for pub in publications:
            try:
                # Clean title
                title = pub.title.strip()
                title = re.sub(r'\s+', ' ', title)  # Normalize whitespace
                title = title.strip('{}"\'')  # Remove surrounding quotes/braces

                # Clean venue name
                venue_name = None
                if pub.venue_name:
                    venue_name = pub.venue_name.strip()
                    venue_name = re.sub(r'\s+', ' ', venue_name)
                    venue_name = venue_name.strip('{}"\'')

                # Clean authors
                authors = []
                if pub.authors:
                    for author in pub.authors:
                        author = author.strip()
                        author = re.sub(r'\s+', ' ', author)
                        author = author.strip('{}"\'')
                        if author:
                            authors.append(author)

                # Clean DOI
                doi = validate_doi(pub.doi) if pub.doi else None

                # Create cleaned publication
                cleaned_pub = PublicationCreate(
                    faculty_id=pub.faculty_id,
                    title=title,
                    abstract=pub.abstract.strip() if pub.abstract else None,
                    publication_type=pub.publication_type,
                    venue_name=venue_name,
                    publisher=pub.publisher.strip() if pub.publisher else None,
                    volume=pub.volume.strip() if pub.volume else None,
                    issue=pub.issue.strip() if pub.issue else None,
                    pages=pub.pages.strip() if pub.pages else None,
                    doi=doi,
                    isbn=pub.isbn.strip() if pub.isbn else None,
                    issn=pub.issn.strip() if pub.issn else None,
                    url=pub.url.strip() if pub.url else None,
                    publication_year=pub.publication_year,
                    publication_date=pub.publication_date,
                    authors=authors,
                    author_count=len(authors),
                    citation_count=pub.citation_count or 0,
                    impact_factor=pub.impact_factor,
                    source_database=pub.source_database,
                    source_metadata=pub.source_metadata or {}
                )

                cleaned.append(cleaned_pub)

            except Exception as e:
                print(f"Error cleaning publication: {e}")
                continue

        return cleaned

    async def _deduplicate_publications(
        self,
        publications: List[PublicationCreate]
    ) -> List[PublicationCreate]:
        """Remove duplicate publications."""
        if len(publications) <= 1:
            return publications

        # Group publications by potential duplicates
        groups = []
        processed_indices = set()

        for i, pub1 in enumerate(publications):
            if i in processed_indices:
                continue

            group = [pub1]
            processed_indices.add(i)

            # Find potential duplicates
            for j, pub2 in enumerate(publications[i+1:], i+1):
                if j in processed_indices:
                    continue

                similarity = await self._calculate_publication_similarity(pub1, pub2)
                if similarity > 0.85:  # High similarity threshold
                    group.append(pub2)
                    processed_indices.add(j)

            groups.append(group)

        # Choose best representative from each group
        deduplicated = []
        for group in groups:
            if len(group) == 1:
                deduplicated.append(group[0])
            else:
                best = await self._choose_best_publication(group)
                deduplicated.append(best)

        return deduplicated

    async def _categorize_publications(
        self,
        publications: List[PublicationCreate]
    ) -> List[PublicationCreate]:
        """Categorize publications by type (journal, conference, etc.)."""
        categorized = []

        for pub in publications:
            # If already categorized with high confidence, keep it
            if pub.publication_type != PublicationType.OTHER:
                confidence = await self._calculate_type_confidence(pub, pub.publication_type)
                if confidence > 0.8:
                    categorized.append(pub)
                    continue

            # Otherwise, determine type
            detected_type, confidence = await self._detect_publication_type(pub)

            # Update publication type
            pub.publication_type = detected_type

            # Store confidence in metadata
            if pub.source_metadata is None:
                pub.source_metadata = {}
            pub.source_metadata['type_confidence'] = confidence

            categorized.append(pub)

        return categorized

    async def _detect_publication_type(
        self,
        publication: PublicationCreate
    ) -> Tuple[PublicationType, float]:
        """Detect publication type with confidence score."""
        scores = {
            PublicationType.JOURNAL: 0.0,
            PublicationType.CONFERENCE: 0.0,
            PublicationType.BOOK: 0.0,
            PublicationType.BOOK_CHAPTER: 0.0,
            PublicationType.THESIS: 0.0,
            PublicationType.TECHNICAL_REPORT: 0.0,
            PublicationType.OTHER: 0.0
        }

        # Analyze venue name patterns
        if publication.venue_name:
            venue_lower = publication.venue_name.lower()

            # Check for conference indicators
            for pattern in self.conference_keywords:
                if pattern in venue_lower:
                    scores[PublicationType.CONFERENCE] += 2.0

            # Check for journal indicators
            for pattern in self.journal_keywords:
                if pattern in venue_lower:
                    scores[PublicationType.JOURNAL] += 2.0

            # Check for specific venue patterns
            for venue_type, patterns in self.venue_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, venue_lower, re.IGNORECASE):
                        if venue_type == 'conference':
                            scores[PublicationType.CONFERENCE] += 1.5
                        elif venue_type == 'journal':
                            scores[PublicationType.JOURNAL] += 1.5
                        elif venue_type == 'book':
                            scores[PublicationType.BOOK] += 1.5

        # Analyze publisher
        if publication.publisher:
            publisher_lower = publication.publisher.lower()

            for publisher_type, patterns in self.publisher_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, publisher_lower, re.IGNORECASE):
                        if publisher_type == 'university':
                            scores[PublicationType.THESIS] += 1.0
                        elif publisher_type == 'technical':
                            scores[PublicationType.TECHNICAL_REPORT] += 1.0
                        elif publisher_type == 'book':
                            scores[PublicationType.BOOK] += 1.0
                            scores[PublicationType.BOOK_CHAPTER] += 1.0

        # Analyze DOI pattern
        if publication.doi:
            # Journal DOIs often have specific patterns
            if re.search(r'10\.\d+/\w+\.\d+', publication.doi):
                scores[PublicationType.JOURNAL] += 1.0

        # Analyze ISSN/ISBN
        if publication.issn:
            scores[PublicationType.JOURNAL] += 1.5
        if publication.isbn:
            scores[PublicationType.BOOK] += 1.0
            scores[PublicationType.BOOK_CHAPTER] += 1.0

        # Analyze title patterns
        title_lower = publication.title.lower()

        # Thesis indicators
        thesis_indicators = ['phd thesis', 'master thesis', 'dissertation', 'doctoral']
        for indicator in thesis_indicators:
            if indicator in title_lower:
                scores[PublicationType.THESIS] += 2.0

        # Technical report indicators
        tech_indicators = ['technical report', 'tech report', 'working paper', 'preprint']
        for indicator in tech_indicators:
            if indicator in title_lower:
                scores[PublicationType.TECHNICAL_REPORT] += 1.5

        # Analyze pages format
        if publication.pages:
            # Conference proceedings often use pp. format
            if re.search(r'pp\.\s*\d+', publication.pages):
                scores[PublicationType.CONFERENCE] += 0.5

        # Determine best type with confidence
        max_score = max(scores.values())
        if max_score == 0:
            return PublicationType.OTHER, 0.0

        best_type = max(scores, key=scores.get)
        confidence = min(max_score / 3.0, 1.0)  # Normalize to 0-1 range

        return best_type, confidence

    async def _calculate_type_confidence(
        self,
        publication: PublicationCreate,
        detected_type: PublicationType
    ) -> float:
        """Calculate confidence in the detected publication type."""
        _, confidence = await self._detect_publication_type(publication)
        return confidence

    async def _match_to_faculty(
        self,
        publications: List[PublicationCreate],
        faculty_mapping: Dict[str, int]
    ) -> List[PublicationCreate]:
        """Match publications to faculty based on author names."""
        matched_publications = []

        for pub in publications:
            if not pub.authors:
                matched_publications.append(pub)
                continue

            best_match_id = None
            best_score = 0.0

            # Try to match each author to faculty
            for author in pub.authors:
                author_normalized = normalize_faculty_name(author)

                for faculty_name, faculty_id in faculty_mapping.items():
                    faculty_normalized = normalize_faculty_name(faculty_name)

                    # Exact match
                    if author_normalized == faculty_normalized:
                        match_score = 1.0
                    # Contains match
                    elif faculty_normalized in author_normalized or author_normalized in faculty_normalized:
                        match_score = 0.7
                    else:
                        match_score = 0.0

                    if match_score > best_score:
                        best_score = match_score
                        best_match_id = faculty_id

            # Update faculty ID if good match found
            if best_score > 0.5:
                pub.faculty_id = best_match_id

                # Store match confidence in metadata
                if pub.source_metadata is None:
                    pub.source_metadata = {}
                pub.source_metadata['faculty_match_score'] = best_score

            matched_publications.append(pub)

        return matched_publications

    async def _calculate_quality_scores(
        self,
        publications: List[PublicationCreate]
    ) -> List[PublicationCreate]:
        """Calculate quality and relevance scores for publications."""
        for pub in publications:
            quality_score = 0.0
            relevance_score = 0.0

            # Quality score components
            if pub.doi:
                quality_score += 1.0
            if pub.abstract and len(pub.abstract) > 100:
                quality_score += 0.5
            if pub.venue_name:
                quality_score += 0.5
            if pub.authors and len(pub.authors) > 0:
                quality_score += 0.3
            if pub.citation_count and pub.citation_count > 0:
                quality_score += min(pub.citation_count / 10.0, 1.0)
            if pub.impact_factor and pub.impact_factor > 0:
                quality_score += min(pub.impact_factor / 5.0, 1.0)

            # Relevance score (placeholder - would depend on faculty interests)
            # For now, use faculty match score if available
            if pub.source_metadata and 'faculty_match_score' in pub.source_metadata:
                relevance_score = pub.source_metadata['faculty_match_score']

            # Normalize scores to 0-1 range
            pub.relevance_score = min(relevance_score, 1.0)
            pub.confidence_score = min(quality_score / 3.0, 1.0)

        return publications

    async def _calculate_publication_similarity(
        self,
        pub1: PublicationCreate,
        pub2: PublicationCreate
    ) -> float:
        """Calculate similarity between two publications."""
        similarity = 0.0

        # Title similarity (40% weight)
        title1 = normalize_faculty_name(pub1.title)
        title2 = normalize_faculty_name(pub2.title)
        title_similarity = SequenceMatcher(None, title1, title2).ratio()
        similarity += title_similarity * 0.4

        # DOI match (50% weight if both have DOI)
        if pub1.doi and pub2.doi:
            if pub1.doi == pub2.doi:
                similarity += 0.5

        # Author similarity (10% weight)
        if pub1.authors and pub2.authors:
            authors1 = set(normalize_faculty_name(author) for author in pub1.authors)
            authors2 = set(normalize_faculty_name(author) for author in pub2.authors)

            if authors1 and authors2:
                intersection = len(authors1 & authors2)
                union = len(authors1 | authors2)
                author_similarity = intersection / union if union > 0 else 0.0
                similarity += author_similarity * 0.1

        # Year match (bonus)
        if pub1.publication_year and pub2.publication_year:
            if pub1.publication_year == pub2.publication_year:
                similarity += 0.1

        return min(similarity, 1.0)

    async def _choose_best_publication(
        self,
        publications: List[PublicationCreate]
    ) -> PublicationCreate:
        """Choose the best publication from a group of duplicates."""
        if len(publications) == 1:
            return publications[0]

        # Score each publication
        best_pub = publications[0]
        best_score = 0

        for pub in publications:
            score = 0

            # Prefer publications with DOI
            if pub.doi:
                score += 10

            # Prefer publications with abstract
            if pub.abstract:
                score += 3

            # Prefer publications with venue name
            if pub.venue_name:
                score += 2

            # Prefer publications with URL
            if pub.url:
                score += 1

            # Prefer publications from more authoritative sources
            source_priority = {
                'crossref': 4,
                'dblp': 3,
                'openalex': 3,
                'google_scholar': 2,
                'bibtex_import': 1
            }
            score += source_priority.get(pub.source_database, 0)

            # Prefer publications with more complete metadata
            metadata_completeness = 0
            if pub.publisher:
                metadata_completeness += 1
            if pub.volume:
                metadata_completeness += 1
            if pub.pages:
                metadata_completeness += 1
            if pub.authors:
                metadata_completeness += 1

            score += metadata_completeness * 0.5

            if score > best_score:
                best_score = score
                best_pub = pub

        return best_pub

    async def _generate_processing_statistics(
        self,
        original_publications: List[PublicationCreate],
        processed_publications: List[PublicationCreate]
    ) -> Dict:
        """Generate statistics about the processing results."""
        # Count by type
        type_counts = defaultdict(int)
        for pub in processed_publications:
            type_counts[pub.publication_type.value] += 1

        # Count by year
        year_counts = defaultdict(int)
        for pub in processed_publications:
            year_counts[pub.publication_year] += 1

        # Quality statistics
        relevance_scores = [pub.relevance_score for pub in processed_publications if pub.relevance_score]
        confidence_scores = [pub.confidence_score for pub in processed_publications if pub.confidence_score]

        stats = {
            'original_count': len(original_publications),
            'processed_count': len(processed_publications),
            'duplicates_removed': len(original_publications) - len(processed_publications),
            'publication_types': dict(type_counts),
            'years_covered': dict(sorted(year_counts.items())),
            'average_relevance_score': np.mean(relevance_scores) if relevance_scores else 0.0,
            'average_confidence_score': np.mean(confidence_scores) if confidence_scores else 0.0,
            'with_doi': sum(1 for pub in processed_publications if pub.doi),
            'with_abstract': sum(1 for pub in processed_publications if pub.abstract),
            'matched_to_faculty': sum(1 for pub in processed_publications if pub.faculty_id > 1),
        }

        return stats

    def _load_venue_patterns(self) -> Dict[str, List[str]]:
        """Load venue name patterns for type detection."""
        return {
            'conference': [
                r'proceedings? of.*conference',
                r'proceedings? of.*workshop',
                r'.*conference.*proceedings?',
                r'.*workshop.*proceedings?',
                r'.*symposium.*',
                r'.*int\'l.*conference',
                r'.*international.*conference',
                r'acm.*conference',
                r'ieee.*conference',
            ],
            'journal': [
                r'.*journal.*',
                r'.*transactions?.*',
                r'.*review.*',
                r'.*letters?.*',
                r'.*magazine.*',
                r'acm.*transactions?',
                r'ieee.*transactions?',
                r'springer.*',
                r'elsevier.*',
            ],
            'book': [
                r'.*book.*',
                r'.*monograph.*',
                r'.*handbook.*',
                r'springer.*book',
                r'cambridge.*press',
                r'oxford.*press',
            ]
        }

    def _load_conference_keywords(self) -> List[str]:
        """Load keywords indicating conference publications."""
        return [
            'conference', 'workshop', 'symposium', 'proceedings',
            'int\'l', 'international', 'poster', 'demo', 'slides'
        ]

    def _load_journal_keywords(self) -> List[str]:
        """Load keywords indicating journal publications."""
        return [
            'journal', 'transactions', 'review', 'letters', 'magazine',
            'periodical', 'quarterly', 'monthly', 'annual'
        ]

    def _load_publisher_patterns(self) -> Dict[str, List[str]]:
        """Load publisher patterns for type detection."""
        return {
            'university': [
                r'university.*press',
                r'.*university.*',
                r'mit.*press',
                r'stanford.*press',
            ],
            'technical': [
                r'technical.*report',
                r'research.*report',
                r'working.*paper',
                r'preprint',
            ],
            'book': [
                r'springer',
                r'elsevier',
                r'wiley',
                r'cambridge.*press',
                r'oxford.*press',
                r'acm.*press',
                r'ieee.*press',
            ]
        }