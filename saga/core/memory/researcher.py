"""
Sovereign Researcher - Global Knowledge Discovery
==================================================

Enables autonomous knowledge gap resolution through guarded
web research with privacy, domain whitelisting, and HITL verification.

Author: ARC SAGA Development Team
Date: December 25, 2025
Status: USMA Phase 6 - Sovereign Researcher
"""

import hashlib
import logging
import re
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import httpx

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# DOMAIN WARDEN - GUARDRAILS
# ═══════════════════════════════════════════════════════════════

class ResearchDomain(str, Enum):
    """Approved research domains."""
    PYTHON_DOCS = "docs.python.org"
    GITHUB = "github.com"
    STACKOVERFLOW = "stackoverflow.com"
    PYPI = "pypi.org"
    READTHEDOCS = "readthedocs.io"
    MDN_WEB = "developer.mozilla.org"
    NPM = "npmjs.com"
    RUST_DOCS = "doc.rust-lang.org"
    GO_DOCS = "pkg.go.dev"


# Default whitelist of approved domains
DEFAULT_WHITELIST = [
    "docs.python.org",
    "github.com",
    "stackoverflow.com",
    "pypi.org",
    "readthedocs.io",
    "developer.mozilla.org",
    "npmjs.com",
    "doc.rust-lang.org",
    "pkg.go.dev",
    "learn.microsoft.com",
    "wiki.archlinux.org",
    "en.wikipedia.org",
]

# Blocked query patterns (non-software topics)
BLOCKED_PATTERNS = [
    r"\bweather\b",
    r"\bnews\b",
    r"\bsports\b",
    r"\bpolitics\b",
    r"\bcelebrity\b",
    r"\brecipe\b",
    r"\bgames?\b(?!\s*(engine|development|programming))",
    r"\bmovie\b",
    r"\bmusic\b(?!\s*(library|api|programming))",
]


@dataclass
class DomainWarden:
    """
    Guards research queries with domain whitelisting and safety filters.

    Ensures:
    - Only approved domains are queried
    - Queries are software/persona-related
    - No sensitive data leaks
    """
    whitelist: list[str] = field(default_factory=lambda: DEFAULT_WHITELIST.copy())
    blocked_patterns: list[str] = field(default_factory=lambda: BLOCKED_PATTERNS.copy())

    def is_domain_allowed(self, url: str) -> bool:
        """Check if URL domain is whitelisted."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            for allowed in self.whitelist:
                if domain == allowed or domain.endswith(f".{allowed}"):
                    return True

            return False
        except Exception:
            return False

    def is_query_allowed(self, query: str) -> tuple[bool, str]:
        """
        Check if query is software/persona-related.

        Returns:
            (allowed, reason) tuple
        """
        query_lower = query.lower()

        for pattern in self.blocked_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return False, "Query blocked: Non-software topic detected"

        return True, "Query approved"

    def add_domain(self, domain: str) -> None:
        """Add a domain to the whitelist."""
        if domain not in self.whitelist:
            self.whitelist.append(domain)
            logger.info(f"Added {domain} to whitelist")

    def remove_domain(self, domain: str) -> bool:
        """Remove a domain from the whitelist."""
        if domain in self.whitelist:
            self.whitelist.remove(domain)
            return True
        return False


# ═══════════════════════════════════════════════════════════════
# QUERY SANITIZER
# ═══════════════════════════════════════════════════════════════

class QuerySanitizer:
    """
    Anonymizes and sanitizes search queries to prevent data leaks.

    Removes:
    - Local file paths
    - API keys and tokens
    - Personal identifiers
    - Code snippets that might contain secrets
    """

    # Patterns to remove from queries
    SANITIZE_PATTERNS = [
        (r"[A-Za-z]:\\[^\s]+", "[LOCAL_PATH]"),  # Windows paths
        (r"/(?:home|Users)/[^\s]+", "[LOCAL_PATH]"),  # Unix paths
        (r"\b[A-Za-z0-9_-]{32,}\b", "[TOKEN]"),  # Long tokens/keys
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]"),
        (r"password\s*[=:]\s*\S+", "password=[REDACTED]"),
        (r"api[_-]?key\s*[=:]\s*\S+", "api_key=[REDACTED]"),
        (r"secret\s*[=:]\s*\S+", "secret=[REDACTED]"),
        (r"token\s*[=:]\s*\S+", "token=[REDACTED]"),
    ]

    def sanitize(self, query: str) -> str:
        """Remove sensitive information from query."""
        sanitized = query

        for pattern, replacement in self.SANITIZE_PATTERNS:
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

        return sanitized.strip()

    def hash_for_cache(self, query: str) -> str:
        """Generate a cache-safe hash for the query."""
        sanitized = self.sanitize(query)
        return hashlib.sha256(sanitized.encode()).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════
# RESEARCH FINDINGS
# ═══════════════════════════════════════════════════════════════

@dataclass
class ResearchFinding:
    """A single research finding from web search."""
    finding_id: str = field(default_factory=lambda: str(uuid4()))
    query: str = ""
    source_url: str = ""
    source_domain: str = ""
    title: str = ""
    content_summary: str = ""
    confidence: float = 0.5
    retrieved_at: datetime = field(default_factory=datetime.utcnow)
    verified: bool = False
    verification_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "query": self.query,
            "source_url": self.source_url,
            "source_domain": self.source_domain,
            "title": self.title,
            "content_summary": self.content_summary,
            "confidence": self.confidence,
            "retrieved_at": self.retrieved_at.isoformat(),
            "verified": self.verified
        }


@dataclass
class FindingsReport:
    """A report of research findings for HITL verification."""
    report_id: str = field(default_factory=lambda: str(uuid4()))
    query: str = ""
    findings: list[ResearchFinding] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    verified: bool = False
    integrated: bool = False

    def render_for_review(self) -> str:
        """Render the report for human review."""
        lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║  🔍 RESEARCH FINDINGS REPORT                                 ║",
            "╚══════════════════════════════════════════════════════════════╝",
            "",
            f"Query: {self.query}",
            f"Findings: {len(self.findings)}",
            f"Generated: {self.created_at.strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "───────────────────────────────────────────────────────────────",
        ]

        for i, finding in enumerate(self.findings, 1):
            lines.extend([
                "",
                f"  [{i}] {finding.title}",
                f"      Source: {finding.source_url}",
                f"      Confidence: {finding.confidence:.0%}",
                f"      Summary: {finding.content_summary[:200]}...",
            ])

        lines.extend([
            "",
            "───────────────────────────────────────────────────────────────",
            "  ⚠️  Please verify sources before integration.",
            "───────────────────────────────────────────────────────────────",
        ])

        return "\n".join(lines)

    def open_sources_in_browser(self) -> int:
        """Open all source URLs in the default browser."""
        count = 0
        for finding in self.findings:
            if finding.source_url:
                try:
                    webbrowser.open(finding.source_url)
                    count += 1
                except Exception:
                    pass
        return count


# ═══════════════════════════════════════════════════════════════
# RESEARCHER AGENTS
# ═══════════════════════════════════════════════════════════════

class ResearcherAgent:
    """
    Base class for research agents.

    Handles common functionality:
    - Domain validation
    - Query sanitization
    - HTTP fetching with timeout
    - Offline fallback
    """

    def __init__(
        self,
        warden: DomainWarden | None = None,
        timeout: float = 10.0,
        offline_mode: bool = False
    ):
        """Initialize the researcher."""
        self.warden = warden or DomainWarden()
        self.sanitizer = QuerySanitizer()
        self.timeout = timeout
        self.offline_mode = offline_mode
        self._cache: dict[str, ResearchFinding] = {}

        logger.info("ResearcherAgent initialized")

    async def fetch_url(self, url: str) -> tuple[str, bool]:
        """
        Fetch content from a URL.

        Returns:
            (content, success) tuple
        """
        if self.offline_mode:
            return "", False

        if not self.warden.is_domain_allowed(url):
            logger.warning(f"Domain not whitelisted: {url}")
            return "", False

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()
                return response.text, True
        except httpx.TimeoutException:
            logger.warning(f"Timeout fetching {url}")
            return "", False
        except httpx.HTTPError as e:
            logger.warning(f"HTTP error fetching {url}: {e}")
            return "", False
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return "", False

    def validate_query(self, query: str) -> tuple[bool, str]:
        """Validate and sanitize a research query."""
        allowed, reason = self.warden.is_query_allowed(query)
        if not allowed:
            return False, reason

        sanitized = self.sanitizer.sanitize(query)
        if not sanitized:
            return False, "Query was empty after sanitization"

        return True, sanitized


class TechnicalResearcher(ResearcherAgent):
    """
    Researches technical documentation for libraries and APIs.

    Specializes in:
    - Python package documentation
    - API references
    - Best practices and patterns
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.research_type = "technical"

    async def research_library(self, library_name: str) -> FindingsReport:
        """
        Research documentation for a library.

        Args:
            library_name: Name of the library to research

        Returns:
            FindingsReport with documentation links
        """
        valid, query = self.validate_query(f"{library_name} python documentation")
        if not valid:
            return FindingsReport(query=library_name, findings=[])

        report = FindingsReport(query=library_name)

        # Try PyPI first
        pypi_url = f"https://pypi.org/project/{library_name}/"
        content, success = await self.fetch_url(pypi_url)

        if success:
            report.findings.append(ResearchFinding(
                query=library_name,
                source_url=pypi_url,
                source_domain="pypi.org",
                title=f"{library_name} on PyPI",
                content_summary=self._extract_summary(content, limit=300),
                confidence=0.9
            ))

        # Try ReadTheDocs
        rtd_url = f"https://{library_name}.readthedocs.io/"
        content, success = await self.fetch_url(rtd_url)

        if success:
            report.findings.append(ResearchFinding(
                query=library_name,
                source_url=rtd_url,
                source_domain="readthedocs.io",
                title=f"{library_name} Documentation",
                content_summary=self._extract_summary(content, limit=300),
                confidence=0.85
            ))

        # Try GitHub
        github_url = f"https://github.com/search?q={library_name}+language%3Apython"
        report.findings.append(ResearchFinding(
            query=library_name,
            source_url=github_url,
            source_domain="github.com",
            title=f"{library_name} on GitHub",
            content_summary="Search GitHub for related repositories",
            confidence=0.7
        ))

        return report

    def _extract_summary(self, html_content: str, limit: int = 200) -> str:
        """Extract text summary from HTML content."""
        # Simple extraction - remove tags
        text = re.sub(r'<[^>]+>', ' ', html_content)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:limit] if len(text) > limit else text


class PersonaResearcher(ResearcherAgent):
    """
    Researches traits and characteristics for narrative personas.

    Specializes in:
    - Character voice traits
    - Speaking patterns
    - Cultural references
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.research_type = "persona"

    async def research_persona(self, persona_name: str) -> FindingsReport:
        """
        Research traits for a character/persona.

        Args:
            persona_name: Name of the persona (e.g., "Bender from Futurama")

        Returns:
            FindingsReport with character traits
        """
        valid, query = self.validate_query(f"{persona_name} character traits quotes")
        if not valid:
            return FindingsReport(query=persona_name, findings=[])

        report = FindingsReport(query=persona_name)

        # Try Wikipedia
        wiki_name = persona_name.replace(" ", "_")
        wiki_url = f"https://en.wikipedia.org/wiki/{wiki_name}"
        content, success = await self.fetch_url(wiki_url)

        if success:
            report.findings.append(ResearchFinding(
                query=persona_name,
                source_url=wiki_url,
                source_domain="en.wikipedia.org",
                title=f"{persona_name} on Wikipedia",
                content_summary=self._extract_traits(content),
                confidence=0.8
            ))

        return report

    def _extract_traits(self, html_content: str) -> str:
        """Extract persona traits from HTML content."""
        text = re.sub(r'<[^>]+>', ' ', html_content)
        text = re.sub(r'\s+', ' ', text).strip()

        # Look for personality/character sections
        personality_match = re.search(
            r'(personality|character|traits?)[:\s]+([^.]+\.)',
            text, re.IGNORECASE
        )

        if personality_match:
            return personality_match.group(2).strip()

        return text[:200]


# ═══════════════════════════════════════════════════════════════
# SOVEREIGN RESEARCHER (ORCHESTRATOR)
# ═══════════════════════════════════════════════════════════════

class SovereignResearcher:
    """
    Main orchestrator for research operations.

    Coordinates:
    - Technical and persona research
    - HITL verification flow
    - Mythos integration
    - Optimizer feedback
    """

    def __init__(
        self,
        project_root: Path | str | None = None,
        warden: DomainWarden | None = None,
        optimizer: Any = None  # SovereignOptimizer
    ):
        """Initialize the Sovereign Researcher."""
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.warden = warden or DomainWarden()
        self.optimizer = optimizer

        self.technical = TechnicalResearcher(warden=self.warden)
        self.persona = PersonaResearcher(warden=self.warden)

        self._reports: list[FindingsReport] = []
        self._verified_findings: list[ResearchFinding] = []

        logger.info("SovereignResearcher initialized")

    async def research_knowledge_gap(
        self,
        query: str,
        research_type: str = "technical"
    ) -> FindingsReport:
        """
        Research a knowledge gap.

        Args:
            query: What to research
            research_type: "technical" or "persona"

        Returns:
            FindingsReport for HITL verification
        """
        if research_type == "technical":
            report = await self.technical.research_library(query)
        elif research_type == "persona":
            report = await self.persona.research_persona(query)
        else:
            report = FindingsReport(query=query)

        self._reports.append(report)
        return report

    def verify_report(self, report: FindingsReport, approved: bool) -> None:
        """
        Mark a report as verified (HITL step).

        Args:
            report: The findings report
            approved: Whether the user approved the findings
        """
        report.verified = True

        if approved:
            self._verified_findings.extend(report.findings)
            logger.info(f"Report {report.report_id} approved with {len(report.findings)} findings")
        else:
            logger.info(f"Report {report.report_id} rejected")

    def integrate_to_mythos(
        self,
        finding: ResearchFinding,
        mythos_library: Any  # MythosLibrary
    ) -> bool:
        """
        Integrate a verified finding into the Mythos.

        Args:
            finding: The verified finding
            mythos_library: Target MythosLibrary

        Returns:
            True if successfully integrated
        """
        if not finding.verified:
            logger.warning("Cannot integrate unverified finding")
            return False

        # Create a pattern from the finding
        from saga.core.memory.mythos import MythosChapter, SolvedPattern

        pattern = SolvedPattern(
            name=f"Global: {finding.title}",
            description=finding.content_summary,
            example=f"Source: {finding.source_url}"
        )

        # Add to or create chapter
        chapter = MythosChapter(
            title="Global Knowledge",
            summary="Patterns discovered through research",
            phase="Research Phase",
            solved_patterns=[pattern]
        )

        mythos_library.add_chapter(chapter)

        # Record optimizer feedback if available
        if self.optimizer:
            context = [hash(finding.source_url) % 256 / 256.0] * 64
            self.optimizer.record_feedback(
                task_id=finding.finding_id,
                context_vector=context,
                retrieval_path=[finding.source_url],
                confidence=finding.confidence,
                success=True
            )

        return True

    def get_offline_fallback(self, mythos: Any) -> str:
        """
        Generate fallback message when offline.

        Args:
            mythos: MythosLibrary to fall back to

        Returns:
            Fallback message
        """
        recent = mythos.get_recent_chapters(1)

        if recent:
            return (
                "🔌 Network unavailable. Falling back to Local Mythos.\n"
                f"   Latest chapter: {recent[0].title}"
            )
        else:
            return "🔌 Network unavailable. No local Mythos available."

    def get_stats(self) -> dict[str, Any]:
        """Get researcher statistics."""
        return {
            "total_reports": len(self._reports),
            "verified_findings": len(self._verified_findings),
            "whitelist_size": len(self.warden.whitelist),
            "offline_mode": self.technical.offline_mode
        }
