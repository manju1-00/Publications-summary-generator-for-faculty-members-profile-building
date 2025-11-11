"""
Academic database crawler service for integrating multiple academic databases.
Integrates DBLP, Crossref, OpenAlex, and Google Scholar APIs.
"""

import asyncio
import aiohttp
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime, timedelta
import json
import re
from urllib.parse import quote, urlencode
import hashlib

from app.models.publication import PublicationCreate, PublicationType
from app.models.faculty import FacultyNameVariation
from app.utils.validation import validate_doi, normalize_faculty_name, generate_name_variations


class DatabaseCrawler:
    """Service for crawling multiple academic databases for faculty publications."""

    def __init__(self):
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration_hours = 24
        self.session = None
        self.api_keys = {
            'crossref': None,  # Crossref doesn't require API key for basic usage
            'openalex': None,  # OpenAlex doesn't require API key
            'google_scholar': None,  # Will use scraping API
        }

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'AcademicPublicationSystem/1.0 (mailto:admin@university.edu)'
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def crawl_faculty_publications(
        self,
        faculty_names: List[str],
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        databases: Optional[List[str]] = None
    ) -> List[PublicationCreate]:
        """
        Crawl multiple databases for faculty publications.

        Args:
            faculty_names: List of faculty names to search for
            year_from: Start year for publication search
            year_to: End year for publication search
            databases: List of databases to query (default: all)

        Returns:
            List of publications found across all databases
        """
        if databases is None:
            databases = ['dblp', 'crossref', 'openalex']

        all_publications = []
        seen_dois = set()
        seen_titles = set()

        # Generate name variations for better matching
        faculty_variations = {}
        for name in faculty_names:
            variations = generate_name_variations(name)
            faculty_variations[name] = variations

        # Create search tasks for all databases
        tasks = []
        for db_name in databases:
            for faculty_name, variations in faculty_variations.items():
                task = self._crawl_database(
                    db_name=db_name,
                    name_variations=variations,
                    original_name=faculty_name,
                    year_from=year_from,
                    year_to=year_to
                )
                tasks.append(task)

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and remove duplicates
        for result in results:
            if isinstance(result, Exception):
                print(f"Crawling error: {result}")
                continue

            for publication in result:
                # Check for duplicates based on DOI
                if publication.doi and publication.doi in seen_dois:
                    continue

                # Check for duplicates based on normalized title
                normalized_title = normalize_faculty_name(publication.title)
                if normalized_title in seen_titles:
                    continue

                # Add to results
                if publication.doi:
                    seen_dois.add(publication.doi)
                seen_titles.add(normalized_title)
                all_publications.append(publication)

        # Deduplicate and score publications
        deduplicated = await self._deduplicate_publications(all_publications)

        return deduplicated

    async def _crawl_database(
        self,
        db_name: str,
        name_variations: List[str],
        original_name: str,
        year_from: Optional[int],
        year_to: Optional[int]
    ) -> List[PublicationCreate]:
        """Crawl a specific database for publications."""
        cache_key = self._generate_cache_key(db_name, name_variations, year_from, year_to)

        # Check cache
        if await self._is_cache_valid(cache_key):
            return self.cache[cache_key]

        try:
            if db_name == 'dblp':
                publications = await self._crawl_dblp(name_variations, original_name, year_from, year_to)
            elif db_name == 'crossref':
                publications = await self._crawl_crossref(name_variations, original_name, year_from, year_to)
            elif db_name == 'openalex':
                publications = await self._crawl_openalex(name_variations, original_name, year_from, year_to)
            elif db_name == 'google_scholar':
                publications = await self._crawl_google_scholar(name_variations, original_name, year_from, year_to)
            else:
                publications = []

            # Cache results
            self.cache[cache_key] = publications
            self.cache_expiry[cache_key] = datetime.now() + timedelta(hours=self.cache_duration_hours)

            return publications

        except Exception as e:
            print(f"Error crawling {db_name}: {e}")
            return []

    async def _crawl_dblp(
        self,
        name_variations: List[str],
        original_name: str,
        year_from: Optional[int],
        year_to: Optional[int]
    ) -> List[PublicationCreate]:
        """Crawl DBLP database for publications."""
        publications = []
        base_url = "https://dblp.org/search/publ/api"

        for name in name_variations[:5]:  # Limit to avoid rate limiting
            try:
                # Build query parameters
                params = {
                    'q': name,
                    'format': 'json',
                    'h': 100,  # Number of results
                    'f': 0     # First result
                }

                url = f"{base_url}?{urlencode(params)}"

                async with self.session.get(url) as response:
                    if response.status != 200:
                        continue

                    data = await response.json()
                    if 'result' not in data or 'hits' not in data['result']:
                        continue

                    hits = data['result']['hits'].get('hit', [])
                    for hit in hits:
                        try:
                            pub_data = hit['info']
                            publication = await self._parse_dblp_entry(pub_data, original_name)
                            if publication:
                                publications.append(publication)
                        except Exception as e:
                            print(f"Error parsing DBLP entry: {e}")
                            continue

                # Rate limiting
                await asyncio.sleep(1)

            except Exception as e:
                print(f"Error querying DBLP for name {name}: {e}")
                continue

        return publications

    async def _crawl_crossref(
        self,
        name_variations: List[str],
        original_name: str,
        year_from: Optional[int],
        year_to: Optional[int]
    ) -> List[PublicationCreate]:
        """Crawl Crossref database for publications."""
        publications = []
        base_url = "https://api.crossref.org/works"

        for name in name_variations[:5]:  # Limit to avoid rate limiting
            try:
                # Build query parameters
                params = {
                    'query.author': name,
                    'rows': 100,
                    'sort': 'published',
                    'order': 'desc'
                }

                if year_from:
                    params['filter'] = f'from-pub-date:{year_from}'
                if year_to:
                    filter_val = params.get('filter', '')
                    if filter_val:
                        params['filter'] = f"{filter_val},until-pub-date:{year_to}"
                    else:
                        params['filter'] = f'until-pub-date:{year_to}'

                url = f"{base_url}?{urlencode(params)}"

                async with self.session.get(url) as response:
                    if response.status != 200:
                        continue

                    data = await response.json()
                    if 'message' not in data or 'items' not in data['message']:
                        continue

                    items = data['message']['items']
                    for item in items:
                        try:
                            publication = await self._parse_crossref_entry(item, original_name)
                            if publication:
                                publications.append(publication)
                        except Exception as e:
                            print(f"Error parsing Crossref entry: {e}")
                            continue

                # Rate limiting
                await asyncio.sleep(1)

            except Exception as e:
                print(f"Error querying Crossref for name {name}: {e}")
                continue

        return publications

    async def _crawl_openalex(
        self,
        name_variations: List[str],
        original_name: str,
        year_from: Optional[int],
        year_to: Optional[int]
    ) -> List[PublicationCreate]:
        """Crawl OpenAlex database for publications."""
        publications = []
        base_url = "https://api.openalex.org/works"

        for name in name_variations[:5]:  # Limit to avoid rate limiting
            try:
                # Build query parameters
                params = {
                    'filter': f'author.display_name.search:"{name}"',
                    'per-page': 100,
                    'sort': 'publication_year:desc'
                }

                if year_from:
                    filter_val = params['filter']
                    params['filter'] = f"{filter_val},publication_year:>={year_from}"
                if year_to:
                    filter_val = params['filter']
                    params['filter'] = f"{filter_val},publication_year:<={year_to}"

                url = f"{base_url}?{urlencode(params)}"

                async with self.session.get(url) as response:
                    if response.status != 200:
                        continue

                    data = await response.json()
                    if 'results' not in data:
                        continue

                    results = data['results']
                    for item in results:
                        try:
                            publication = await self._parse_openalex_entry(item, original_name)
                            if publication:
                                publications.append(publication)
                        except Exception as e:
                            print(f"Error parsing OpenAlex entry: {e}")
                            continue

                # Rate limiting (OpenAlex is more generous)
                await asyncio.sleep(0.1)

            except Exception as e:
                print(f"Error querying OpenAlex for name {name}: {e}")
                continue

        return publications

    async def _crawl_google_scholar(
        self,
        name_variations: List[str],
        original_name: str,
        year_from: Optional[int],
        year_to: Optional[int]
    ) -> List[PublicationCreate]:
        """Crawl Google Scholar using scraping API."""
        # Note: This would require integration with a scraping service like ScrapingBee
        # For now, return empty list as placeholder
        print("Google Scholar crawling not implemented - requires scraping API integration")
        return []

    async def _parse_dblp_entry(self, entry: Dict, original_name: str) -> Optional[PublicationCreate]:
        """Parse DBLP entry into publication record."""
        try:
            title = entry.get('title', {}).get('$', '')
            if not title:
                return None

            # Determine publication type from venue type
            venue_type = entry.get('type', '')
            if venue_type == 'Journal Articles':
                pub_type = PublicationType.JOURNAL
            elif venue_type in ['Conference and Workshop Papers', 'Informal Publications']:
                pub_type = PublicationType.CONFERENCE
            elif venue_type == 'Books and Theses':
                pub_type = PublicationType.BOOK if entry.get('type') == 'Book' else PublicationType.THESIS
            else:
                pub_type = PublicationType.OTHER

            # Extract authors
            authors = []
            if 'authors' in entry and 'author' in entry['authors']:
                author_list = entry['authors']['author']
                if isinstance(author_list, dict):
                    author_list = [author_list]
                for author in author_list:
                    if isinstance(author, dict) and '$' in author:
                        authors.append(author['$'])

            # Extract year
            year = None
            if 'year' in entry:
                try:
                    year = int(entry['year'])
                except (ValueError, TypeError):
                    pass

            # Extract venue
            venue = None
            if 'venue' in entry:
                venue = entry['venue'].get('$', '')

            # Extract DOI
            doi = None
            if 'ee' in entry:
                doi = entry['ee'].get('$', '')
                doi = validate_doi(doi)

            # Extract URL
            url = None
            if 'url' in entry:
                url = entry['url'].get('$', '')

            # Extract pages
            pages = None
            if 'pages' in entry:
                pages = entry['pages'].get('$', '')

            return PublicationCreate(
                faculty_id=1,  # Will be updated after faculty matching
                title=title,
                publication_type=pub_type,
                venue_name=venue,
                publication_year=year or datetime.now().year,
                authors=authors,
                author_count=len(authors),
                doi=doi,
                url=url,
                pages=pages,
                source_database="dblp",
                source_metadata={
                    'dblp_key': entry.get('@id', ''),
                    'raw_entry': entry
                }
            )

        except Exception as e:
            print(f"Error parsing DBLP entry: {e}")
            return None

    async def _parse_crossref_entry(self, entry: Dict, original_name: str) -> Optional[PublicationCreate]:
        """Parse Crossref entry into publication record."""
        try:
            title = ' '.join(entry.get('title', []))
            if not title:
                return None

            # Determine publication type
            pub_type_str = entry.get('type', 'journal-article')
            if pub_type_str == 'journal-article':
                pub_type = PublicationType.JOURNAL
            elif pub_type_str in ['proceedings-article', 'conference-paper']:
                pub_type = PublicationType.CONFERENCE
            elif pub_type_str in ['book', 'monograph']:
                pub_type = PublicationType.BOOK
            elif pub_type_str == 'book-chapter':
                pub_type = PublicationType.BOOK_CHAPTER
            else:
                pub_type = PublicationType.OTHER

            # Extract authors
            authors = []
            if 'author' in entry:
                for author in entry['author']:
                    if 'given' in author and 'family' in author:
                        authors.append(f"{author['given']} {author['family']}")
                    elif 'name' in author:
                        authors.append(author['name'])

            # Extract publication date
            year = None
            if 'published-print' in entry and 'date-parts' in entry['published-print']:
                year = entry['published-print']['date-parts'][0][0]
            elif 'issued' in entry and 'date-parts' in entry['issued']:
                year = entry['issued']['date-parts'][0][0]

            # Extract venue
            venue = None
            if 'short-container-title' in entry:
                venue = ' '.join(entry['short-container-title'])
            elif 'container-title' in entry:
                venue = ' '.join(entry['container-title'])

            # Extract DOI
            doi = entry.get('DOI', '')
            doi = validate_doi(doi)

            # Extract URL
            url = entry.get('URL', '')

            # Extract abstract
            abstract = entry.get('abstract', '')

            return PublicationCreate(
                faculty_id=1,  # Will be updated after faculty matching
                title=title,
                abstract=abstract,
                publication_type=pub_type,
                venue_name=venue,
                publication_year=year or datetime.now().year,
                authors=authors,
                author_count=len(authors),
                doi=doi,
                url=url,
                source_database="crossref",
                source_metadata={
                    'crossref_doi': doi,
                    'raw_entry': entry
                }
            )

        except Exception as e:
            print(f"Error parsing Crossref entry: {e}")
            return None

    async def _parse_openalex_entry(self, entry: Dict, original_name: str) -> Optional[PublicationCreate]:
        """Parse OpenAlex entry into publication record."""
        try:
            title = entry.get('title', '')
            if not title:
                return None

            # Determine publication type
            pub_type_str = entry.get('type', 'journal-article')
            if pub_type_str == 'journal-article':
                pub_type = PublicationType.JOURNAL
            elif pub_type_str in ['conference-paper', 'proceedings']:
                pub_type = PublicationType.CONFERENCE
            elif pub_type_str in ['book', 'monograph']:
                pub_type = PublicationType.BOOK
            elif pub_type_str == 'book-chapter':
                pub_type = PublicationType.BOOK_CHAPTER
            else:
                pub_type = PublicationType.OTHER

            # Extract authors
            authors = []
            if 'authorships' in entry:
                for authorship in entry['authorships']:
                    if 'author' in authorship and 'display_name' in authorship['author']:
                        authors.append(authorship['author']['display_name'])

            # Extract year
            year = entry.get('publication_year')

            # Extract venue
            venue = None
            if 'primary_location' in entry and 'source' in entry['primary_location']:
                venue = entry['primary_location']['source'].get('display_name')

            # Extract DOI
            doi = None
            if 'primary_location' in entry and entry['primary_location'].get('doi'):
                doi = entry['primary_location']['doi']
                doi = validate_doi(doi)

            # Extract URL
            url = None
            if 'primary_location' in entry and entry['primary_location'].get('landing_page_url'):
                url = entry['primary_location']['landing_page_url']

            # Extract abstract
            abstract = entry.get('abstract', '')

            # Extract citation count
            citation_count = entry.get('cited_by_count', 0)

            return PublicationCreate(
                faculty_id=1,  # Will be updated after faculty matching
                title=title,
                abstract=abstract,
                publication_type=pub_type,
                venue_name=venue,
                publication_year=year or datetime.now().year,
                authors=authors,
                author_count=len(authors),
                doi=doi,
                url=url,
                citation_count=citation_count,
                source_database="openalex",
                source_metadata={
                    'openalex_id': entry.get('id', ''),
                    'raw_entry': entry
                }
            )

        except Exception as e:
            print(f"Error parsing OpenAlex entry: {e}")
            return None

    async def _deduplicate_publications(self, publications: List[PublicationCreate]) -> List[PublicationCreate]:
        """Remove duplicate publications and apply confidence scoring."""
        if not publications:
            return []

        # Group by title similarity
        groups = []
        used_indices = set()

        for i, pub1 in enumerate(publications):
            if i in used_indices:
                continue

            group = [pub1]
            used_indices.add(i)

            for j, pub2 in enumerate(publications[i+1:], i+1):
                if j in used_indices:
                    continue

                # Check for similarity
                similarity = await self._calculate_similarity(pub1, pub2)
                if similarity > 0.8:  # High similarity threshold
                    group.append(pub2)
                    used_indices.add(j)

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

    async def _calculate_similarity(self, pub1: PublicationCreate, pub2: PublicationCreate) -> float:
        """Calculate similarity between two publications."""
        similarity = 0.0

        # Title similarity
        title1 = normalize_faculty_name(pub1.title)
        title2 = normalize_faculty_name(pub2.title)

        # Simple Jaccard similarity on words
        words1 = set(title1.split())
        words2 = set(title2.split())
        if words1 and words2:
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            title_similarity = intersection / union
            similarity += title_similarity * 0.4

        # DOI match
        if pub1.doi and pub2.doi and pub1.doi == pub2.doi:
            similarity += 0.5

        # Author overlap
        if pub1.authors and pub2.authors:
            authors1 = set(normalize_faculty_name(author) for author in pub1.authors)
            authors2 = set(normalize_faculty_name(author) for author in pub2.authors)
            if authors1 and authors2:
                author_similarity = len(authors1 & authors2) / len(authors1 | authors2)
                similarity += author_similarity * 0.3

        # Year match
        if pub1.publication_year and pub2.publication_year:
            if pub1.publication_year == pub2.publication_year:
                similarity += 0.2

        return min(similarity, 1.0)

    async def _choose_best_publication(self, publications: List[PublicationCreate]) -> PublicationCreate:
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

            # Prefer publications with more complete metadata
            if pub.abstract:
                score += 2
            if pub.venue_name:
                score += 2
            if pub.url:
                score += 1

            # Prefer publications from more authoritative sources
            source_priority = {
                'crossref': 3,
                'dblp': 2,
                'openalex': 2,
                'google_scholar': 1
            }
            score += source_priority.get(pub.source_database, 0)

            if score > best_score:
                best_score = score
                best_pub = pub

        return best_pub

    def _generate_cache_key(
        self,
        db_name: str,
        name_variations: List[str],
        year_from: Optional[int],
        year_to: Optional[int]
    ) -> str:
        """Generate cache key for search results."""
        key_data = {
            'db': db_name,
            'names': sorted(name_variations),
            'year_from': year_from,
            'year_to': year_to
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    async def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid."""
        if cache_key not in self.cache:
            return False

        if cache_key not in self.cache_expiry:
            return False

        return datetime.now() < self.cache_expiry[cache_key]

    async def clear_cache(self):
        """Clear all cached results."""
        self.cache.clear()
        self.cache_expiry.clear()