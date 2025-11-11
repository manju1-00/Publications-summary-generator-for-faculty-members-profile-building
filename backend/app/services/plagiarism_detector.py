"""
Plagiarism detection service with hybrid cost-effective approach.
Combines BERT-based semantic similarity with EdenAI for high-value publications.
"""

import asyncio
import aiohttp
import json
import re
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict

from app.models.plagiarism import (
    PlagiarismScan, PlagiarismSeverity, PlagiarismDetectionEngine,
    PlagiarismStatus, PlagiarismSourceMatch, PlagiarismSentenceMatch
)
from app.models.publication import Publication


class PlagiarismDetector:
    """Service for detecting plagiarism in academic publications."""

    def __init__(self):
        # Load BERT model for semantic similarity
        self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')

        # EdenAI API configuration
        self.edenai_api_key = None  # Set from environment
        self.edenai_base_url = "https://api.edenai.run/v2"

        # Detection thresholds
        self.similarity_threshold_low = 0.3
        self.similarity_threshold_medium = 0.5
        self.similarity_threshold_high = 0.7
        self.similarity_threshold_critical = 0.9

        # Source data for comparison (placeholder - would be actual academic database)
        self.reference_texts = []
        self.reference_metadata = []

        # Cache for processed texts
        self.embedding_cache = {}

    async def scan_publication(
        self,
        publication: Publication,
        scan_config: Optional[Dict] = None
    ) -> PlagiarismScan:
        """
        Scan a publication for plagiarism.

        Args:
            publication: Publication to scan
            scan_config: Configuration for scanning

        Returns:
            PlagiarismScan with results
        """
        if scan_config is None:
            scan_config = self._get_default_scan_config()

        # Create scan record
        scan = PlagiarismScan(
            publication_id=publication.id,
            engine=PlagiarismDetectionEngine.BERT_SEMANTIC,
            scan_version="1.0",
            check_against_internet=scan_config.get('check_internet', True),
            check_against_academic=scan_config.get('check_academic', True),
            check_against_internal=scan_config.get('check_internal', True),
            exclude_quotes=scan_config.get('exclude_quotes', True),
            exclude_references=scan_config.get('exclude_references', True),
            scan_parameters=scan_config
        )

        try:
            # Extract text to analyze
            text_to_scan = await self._extract_text(publication, scan_config)

            if not text_to_scan or len(text_to_scan.strip()) < 100:
                scan.status = PlagiarismStatus.COMPLETED
                scan.overall_similarity_score = 0.0
                scan.max_similarity_score = 0.0
                scan.severity_level = PlagiarismSeverity.LOW
                scan.processing_time_seconds = 0.0
                return scan

            start_time = datetime.now()

            # Phase 1: BERT-based semantic similarity detection
            bert_results = await self._bert_similarity_detection(text_to_scan, scan_config)

            # Calculate overall scores
            overall_similarity = self._calculate_overall_similarity(bert_results)
            max_similarity = max([r['similarity'] for r in bert_results], default=0.0)
            severity = self._determine_severity(overall_similarity, max_similarity)

            # Phase 2: Trigger EdenAI for high-value publications if needed
            edenai_results = []
            if self._should_trigger_edenai(publication, overall_similarity):
                edenai_results = await self._edenai_detection(text_to_scan, scan_config)
                # Combine results
                bert_results.extend(edenai_results)

            # Generate sentence-level matches
            sentence_matches = await self._generate_sentence_matches(text_to_scan, bert_results)

            # Generate source matches
            source_matches = await self._generate_source_matches(bert_results)

            # Update scan with results
            scan.overall_similarity_score = overall_similarity
            scan.max_similarity_score = max_similarity
            scan.severity_level = severity
            scan.status = PlagiarismStatus.COMPLETED
            scan.source_matches = [match.dict() for match in source_matches]
            scan.sentence_matches = [match.dict() for match in sentence_matches]
            scan.processing_time_seconds = (datetime.now() - start_time).total_seconds()
            scan.confidence_score = self._calculate_confidence_score(bert_results)

            # Flag for manual review if needed
            if overall_similarity > self.similarity_threshold_medium:
                scan.status = PlagiarismStatus.FLAGGED

        except Exception as e:
            scan.status = PlagiarismStatus.FAILED
            scan.error_message = str(e)

        return scan

    async def batch_scan_publications(
        self,
        publications: List[Publication],
        batch_config: Optional[Dict] = None
    ) -> List[PlagiarismScan]:
        """
        Scan multiple publications in batch.

        Args:
            publications: List of publications to scan
            batch_config: Batch configuration

        Returns:
            List of plagiarism scans
        """
        if batch_config is None:
            batch_config = {
                'batch_size': 10,
                'priority': 'medium',
                'secondary_scan_threshold': 0.3
            }

        scans = []
        batch_size = batch_config['batch_size']

        # Process in batches to manage memory and API limits
        for i in range(0, len(publications), batch_size):
            batch = publications[i:i + batch_size]

            # Process batch concurrently
            batch_tasks = []
            for pub in batch:
                task = self.scan_publication(pub)
                batch_tasks.append(task)

            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Filter out exceptions and add valid results
            for result in batch_results:
                if isinstance(result, Exception):
                    print(f"Error in batch scan: {result}")
                    continue
                scans.append(result)

        return scans

    async def _extract_text(
        self,
        publication: Publication,
        scan_config: Dict
    ) -> str:
        """Extract text from publication for analysis."""
        text_parts = []

        # Use abstract if available
        if publication.abstract:
            text_parts.append(publication.abstract)

        # Use title
        if publication.title:
            text_parts.append(publication.title)

        # For now, combine title and abstract
        # In a real implementation, you would fetch the full text from DOI or URL
        full_text = " ".join(text_parts)

        # Preprocess text
        full_text = self._preprocess_text(full_text, scan_config)

        return full_text

    def _preprocess_text(self, text: str, scan_config: Dict) -> str:
        """Preprocess text for plagiarism detection."""
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove citations and references if configured
        if scan_config.get('exclude_references', True):
            # Remove common citation patterns
            text = re.sub(r'\[\d+\]', '', text)  # [1], [2], etc.
            text = re.sub(r'\([^)]*\d{4}[^)]*\)', '', text)  # (Author, 2020)
            text = re.sub(r'et al\.', '', text)

        # Remove quotes if configured
        if scan_config.get('exclude_quotes', True):
            text = re.sub(r'["""](.*?)["""]', r'\1', text)
            text = re.sub(r'\'([^\']*)\'', r'\1', text)

        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)

        # Remove extra characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.,!?;:()-]', '', text)

        # Normalize whitespace again
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    async def _bert_similarity_detection(
        self,
        text: str,
        scan_config: Dict
    ) -> List[Dict]:
        """Perform BERT-based semantic similarity detection."""
        # Split text into chunks for processing
        text_chunks = self._split_text_into_chunks(text)

        # Generate embeddings for input text
        text_embeddings = []
        for chunk in text_chunks:
            embedding = await self._get_embedding(chunk)
            text_embeddings.append(embedding)

        # Compare with reference texts
        matches = []

        # For demo purposes, create some reference texts
        # In production, this would be a large academic database
        reference_texts = await self._get_reference_texts(scan_config)

        for i, chunk in enumerate(text_chunks):
            chunk_embedding = text_embeddings[i]

            for ref_text, ref_metadata in reference_texts:
                # Split reference text into chunks
                ref_chunks = self._split_text_into_chunks(ref_text)

                for ref_chunk in ref_chunks:
                    ref_embedding = await self._get_embedding(ref_chunk)

                    # Calculate similarity
                    similarity = cosine_similarity(
                        [chunk_embedding],
                        [ref_embedding]
                    )[0][0]

                    if similarity > self.similarity_threshold_low:
                        matches.append({
                            'text_chunk': chunk,
                            'reference_chunk': ref_chunk,
                            'similarity': float(similarity),
                            'reference_metadata': ref_metadata,
                            'chunk_index': i
                        })

        # Sort by similarity and keep top matches
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        return matches[:50]  # Keep top 50 matches

    async def _edenai_detection(
        self,
        text: str,
        scan_config: Dict
    ) -> List[Dict]:
        """Perform EdenAI plagiarism detection for high-value publications."""
        if not self.edenai_api_key:
            print("EdenAI API key not configured")
            return []

        try:
            headers = {
                'Authorization': f'Bearer {self.edenai_api_key}',
                'Content-Type': 'application/json'
            }

            data = {
                'providers': ['openai', 'copyleaks'],
                'text': text,
                'language': 'en'
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.edenai_base_url}/text/plagiarism",
                    headers=headers,
                    json=data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return await self._parse_edenai_results(result)
                    else:
                        print(f"EdenAI API error: {response.status}")
                        return []

        except Exception as e:
            print(f"EdenAI detection error: {e}")
            return []

    async def _parse_edenai_results(self, edenai_result: Dict) -> List[Dict]:
        """Parse results from EdenAI API."""
        matches = []

        # Parse OpenAI results
        if 'openai' in edenai_result:
            openai_data = edenai_result['openai']
            if 'plagiarism_score' in openai_data:
                score = openai_data['plagiarism_score']
                if score > self.similarity_threshold_low:
                    matches.append({
                        'text_chunk': openai_data.get('text', ''),
                        'reference_chunk': openai_data.get('source_text', ''),
                        'similarity': score,
                        'reference_metadata': {
                            'source': 'openai',
                            'url': openai_data.get('source_url', '')
                        },
                        'chunk_index': 0
                    })

        # Parse Copyleaks results
        if 'copyleaks' in edenai_result:
            copyleaks_data = edenai_result['copyleaks']
            for result in copyleaks_data.get('results', []):
                similarity = result.get('similarity', 0)
                if similarity > self.similarity_threshold_low:
                    matches.append({
                        'text_chunk': result.get('text', ''),
                        'reference_chunk': result.get('matched_text', ''),
                        'similarity': similarity,
                        'reference_metadata': {
                            'source': 'copyleaks',
                            'url': result.get('url', ''),
                            'title': result.get('title', '')
                        },
                        'chunk_index': 0
                    })

        return matches

    async def _generate_sentence_matches(
        self,
        text: str,
        similarity_results: List[Dict]
    ) -> List[PlagiarismSentenceMatch]:
        """Generate sentence-level plagiarism matches."""
        if not text:
            return []

        # Tokenize text into sentences
        try:
            sentences = sent_tokenize(text)
        except:
            # Fallback if NLTK tokenization fails
            sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]

        sentence_matches = []

        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) < 10:  # Skip very short sentences
                continue

            # Find matches for this sentence
            sentence_sources = []
            max_similarity = 0.0

            for match in similarity_results:
                if sentence.lower() in match['text_chunk'].lower():
                    similarity = match['similarity']
                    if similarity > max_similarity:
                        max_similarity = similarity

                    # Create source match
                    source_match = PlagiarismSourceMatch(
                        source_title=match['reference_metadata'].get('title', 'Unknown'),
                        source_url=match['reference_metadata'].get('url', ''),
                        source_author=match['reference_metadata'].get('author', ''),
                        similarity_score=similarity,
                        matched_text=match['reference_chunk'],
                        matched_length=len(match['reference_chunk']),
                        start_position=0,
                        end_position=len(sentence),
                        match_type="semantic"
                    )
                    sentence_sources.append(source_match)

            if max_similarity > self.similarity_threshold_low:
                sentence_match = PlagiarismSentenceMatch(
                    sentence_text=sentence,
                    sentence_index=i,
                    similarity_score=max_similarity,
                    matching_sources=sentence_sources
                )
                sentence_matches.append(sentence_match)

        return sentence_matches

    async def _generate_source_matches(
        self,
        similarity_results: List[Dict]
    ) -> List[PlagiarismSourceMatch]:
        """Generate source-level plagiarism matches."""
        source_matches = []
        seen_sources = set()

        for match in similarity_results:
            metadata = match['reference_metadata']
            source_key = (metadata.get('title', ''), metadata.get('url', ''))

            if source_key in seen_sources:
                continue

            seen_sources.add(source_key)

            source_match = PlagiarismSourceMatch(
                source_title=metadata.get('title', 'Unknown Source'),
                source_url=metadata.get('url', ''),
                source_author=metadata.get('author', ''),
                similarity_score=match['similarity'],
                matched_text=match['reference_chunk'],
                matched_length=len(match['reference_chunk']),
                start_position=0,
                end_position=len(match['text_chunk']),
                match_type="semantic"
            )
            source_matches.append(source_match)

        return source_matches

    def _split_text_into_chunks(self, text: str, chunk_size: int = 250) -> List[str]:
        """Split text into chunks for processing."""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        sentences = sent_tokenize(text)
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get BERT embedding for text with caching."""
        # Check cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]

        # Generate embedding
        embedding = self.bert_model.encode(text)

        # Cache the result
        self.embedding_cache[text_hash] = embedding

        return embedding

    async def _get_reference_texts(self, scan_config: Dict) -> List[Tuple[str, Dict]]:
        """Get reference texts for comparison."""
        # In a real implementation, this would query academic databases
        # For now, return some sample reference texts

        reference_texts = [
            ("This paper presents a novel approach to machine learning classification.",
             {"title": "Sample Paper 1", "author": "Author 1", "year": 2020}),
            ("The experimental results demonstrate significant improvement over baseline methods.",
             {"title": "Sample Paper 2", "author": "Author 2", "year": 2021}),
            ("We propose a new algorithm for natural language processing tasks.",
             {"title": "Sample Paper 3", "author": "Author 3", "year": 2019}),
        ]

        return reference_texts

    def _calculate_overall_similarity(self, similarity_results: List[Dict]) -> float:
        """Calculate overall similarity score."""
        if not similarity_results:
            return 0.0

        # Use weighted average of top matches
        similarities = [match['similarity'] for match in similarity_results[:10]]

        if not similarities:
            return 0.0

        # Weight recent and high-similarity matches more heavily
        weights = []
        for i, similarity in enumerate(similarities):
            weight = (1.0 - i * 0.1) * similarity  # Decreasing weight with position
            weights.append(max(weight, 0.1))

        weighted_sum = sum(s * w for s, w in zip(similarities, weights))
        total_weight = sum(weights)

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _determine_severity(self, overall_similarity: float, max_similarity: float) -> PlagiarismSeverity:
        """Determine plagiarism severity level."""
        # Use maximum similarity for severity determination
        if max_similarity >= self.similarity_threshold_critical:
            return PlagiarismSeverity.CRITICAL
        elif max_similarity >= self.similarity_threshold_high:
            return PlagiarismSeverity.HIGH
        elif max_similarity >= self.similarity_threshold_medium:
            return PlagiarismSeverity.MEDIUM
        else:
            return PlagiarismSeverity.LOW

    def _calculate_confidence_score(self, similarity_results: List[Dict]) -> float:
        """Calculate confidence score in the plagiarism detection."""
        if not similarity_results:
            return 0.0

        # Confidence based on number of matches and their consistency
        high_confidence_matches = sum(
            1 for match in similarity_results
            if match['similarity'] > self.similarity_threshold_high
        )

        total_matches = len(similarity_results)

        if total_matches == 0:
            return 0.0

        # Higher confidence with more high-similarity matches
        confidence = min(high_confidence_matches / 5.0, 1.0)  # Cap at 1.0

        # Adjust based on total number of matches
        if total_matches > 10:
            confidence += 0.1
        elif total_matches > 5:
            confidence += 0.05

        return min(confidence, 1.0)

    def _should_trigger_edenai(self, publication: Publication, similarity_score: float) -> bool:
        """Determine if EdenAI detection should be triggered."""
        # Trigger for high-value publications or high similarity scores
        high_similarity = similarity_score > self.similarity_threshold_medium
        high_impact = publication.impact_factor and publication.impact_factor > 2.0
        high_citations = publication.citation_count and publication.citation_count > 50

        return high_similarity or high_impact or high_citations

    def _get_default_scan_config(self) -> Dict:
        """Get default plagiarism scan configuration."""
        return {
            'check_internet': True,
            'check_academic': True,
            'check_internal': True,
            'exclude_quotes': True,
            'exclude_references': True,
            'similarity_threshold': 0.3,
            'max_results': 50,
            'enable_edenai': True,
            'edenai_threshold': 0.5
        }

    async def get_scan_statistics(self, scans: List[PlagiarismScan]) -> Dict:
        """Get statistics for a batch of plagiarism scans."""
        if not scans:
            return {}

        total_scans = len(scans)
        completed_scans = sum(1 for scan in scans if scan.status == PlagiarismStatus.COMPLETED)
        flagged_scans = sum(1 for scan in scans if scan.status == PlagiarismStatus.FLAGGED)
        failed_scans = sum(1 for scan in scans if scan.status == PlagiarismStatus.FAILED)

        # Calculate similarity statistics
        similarities = [scan.overall_similarity_score for scan in scans if scan.overall_similarity_score is not None]
        avg_similarity = np.mean(similarities) if similarities else 0.0
        max_similarity = max(similarities) if similarities else 0.0

        # Severity breakdown
        severity_counts = defaultdict(int)
        for scan in scans:
            severity_counts[scan.severity_level] += 1

        stats = {
            'total_scans': total_scans,
            'completed_scans': completed_scans,
            'flagged_scans': flagged_scans,
            'failed_scans': failed_scans,
            'average_similarity': float(avg_similarity),
            'max_similarity': float(max_similarity),
            'severity_breakdown': dict(severity_counts),
            'completion_rate': completed_scans / total_scans if total_scans > 0 else 0.0
        }

        return stats