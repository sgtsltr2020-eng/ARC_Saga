"""
Unit Tests for USMA Phase 6: Sovereign Researcher
===================================================

Tests for DomainWarden, QuerySanitizer, Researchers, and HITL flow.
"""


from saga.core.memory import (
    DomainWarden,
    FindingsReport,
    PersonaResearcher,
    QuerySanitizer,
    ResearchFinding,
    SovereignResearcher,
    TechnicalResearcher,
)


class TestDomainWarden:
    """Tests for DomainWarden guardrails."""

    def test_warden_initialization(self):
        """Test warden initializes with default whitelist."""
        warden = DomainWarden()

        assert len(warden.whitelist) > 0
        assert "github.com" in warden.whitelist

    def test_allowed_domain(self):
        """Test whitelisted domain is allowed."""
        warden = DomainWarden()

        assert warden.is_domain_allowed("https://github.com/user/repo") is True
        assert warden.is_domain_allowed("https://docs.python.org/3/") is True
        assert warden.is_domain_allowed("https://pypi.org/project/requests/") is True

    def test_blocked_domain(self):
        """Test non-whitelisted domain is blocked."""
        warden = DomainWarden()

        assert warden.is_domain_allowed("https://malicious-site.com/") is False
        assert warden.is_domain_allowed("https://random-blog.io/") is False

    def test_subdomain_allowed(self):
        """Test subdomains of whitelisted domains are allowed."""
        warden = DomainWarden()

        assert warden.is_domain_allowed("https://requests.readthedocs.io/") is True

    def test_software_query_allowed(self):
        """Test software-related queries are allowed."""
        warden = DomainWarden()

        allowed, _ = warden.is_query_allowed("python requests library documentation")
        assert allowed is True

    def test_non_software_query_blocked(self):
        """Test non-software queries are blocked."""
        warden = DomainWarden()

        allowed, reason = warden.is_query_allowed("what's the weather today")
        assert allowed is False
        assert "Non-software" in reason

        allowed, _ = warden.is_query_allowed("latest celebrity news")
        assert allowed is False

    def test_add_domain(self):
        """Test adding a domain to whitelist."""
        warden = DomainWarden()
        initial_count = len(warden.whitelist)

        warden.add_domain("custom-docs.example.com")

        assert len(warden.whitelist) == initial_count + 1
        assert warden.is_domain_allowed("https://custom-docs.example.com/api")


class TestQuerySanitizer:
    """Tests for QuerySanitizer."""

    def test_sanitizer_initialization(self):
        """Test sanitizer initializes."""
        sanitizer = QuerySanitizer()
        assert sanitizer is not None

    def test_sanitize_windows_path(self):
        """Test Windows paths are sanitized."""
        sanitizer = QuerySanitizer()

        query = "error in C:\\Users\\john\\project\\main.py"
        result = sanitizer.sanitize(query)

        assert "C:\\Users" not in result
        assert "[LOCAL_PATH]" in result

    def test_sanitize_unix_path(self):
        """Test Unix paths are sanitized."""
        sanitizer = QuerySanitizer()

        query = "error in /home/user/project/main.py"
        result = sanitizer.sanitize(query)

        assert "/home/user" not in result

    def test_sanitize_api_key(self):
        """Test API keys are sanitized."""
        sanitizer = QuerySanitizer()

        query = "api_key=test_fake_key_abc123def456ghi789"
        result = sanitizer.sanitize(query)

        assert "test_fake_key" not in result
        assert "[REDACTED]" in result

    def test_sanitize_email(self):
        """Test emails are sanitized."""
        sanitizer = QuerySanitizer()

        query = "contact john.doe@example.com for help"
        result = sanitizer.sanitize(query)

        assert "john.doe@example.com" not in result
        assert "[EMAIL]" in result

    def test_cache_hash(self):
        """Test cache hash generation."""
        sanitizer = QuerySanitizer()

        hash1 = sanitizer.hash_for_cache("python requests")
        hash2 = sanitizer.hash_for_cache("python requests")
        hash3 = sanitizer.hash_for_cache("python asyncio")

        assert hash1 == hash2  # Same query = same hash
        assert hash1 != hash3  # Different query = different hash
        assert len(hash1) == 16


class TestResearchFinding:
    """Tests for ResearchFinding dataclass."""

    def test_finding_creation(self):
        """Test creating a research finding."""
        finding = ResearchFinding(
            query="requests library",
            source_url="https://pypi.org/project/requests/",
            source_domain="pypi.org",
            title="Requests on PyPI",
            content_summary="HTTP library for Python",
            confidence=0.9
        )

        assert finding.query == "requests library"
        assert finding.confidence == 0.9
        assert not finding.verified

    def test_finding_serialization(self):
        """Test finding to_dict."""
        finding = ResearchFinding(
            query="test",
            source_url="https://example.com",
            title="Test Finding"
        )

        data = finding.to_dict()

        assert "finding_id" in data
        assert data["query"] == "test"


class TestFindingsReport:
    """Tests for FindingsReport."""

    def test_report_creation(self):
        """Test creating a findings report."""
        report = FindingsReport(
            query="pytest documentation",
            findings=[
                ResearchFinding(
                    query="pytest",
                    source_url="https://docs.pytest.org/",
                    title="Pytest Docs"
                )
            ]
        )

        assert report.query == "pytest documentation"
        assert len(report.findings) == 1
        assert not report.verified

    def test_report_render(self):
        """Test rendering report for review."""
        report = FindingsReport(
            query="requests library",
            findings=[
                ResearchFinding(
                    query="requests",
                    source_url="https://pypi.org/project/requests/",
                    title="Requests on PyPI",
                    content_summary="HTTP for Humans",
                    confidence=0.9
                )
            ]
        )

        output = report.render_for_review()

        assert "RESEARCH FINDINGS REPORT" in output
        assert "requests" in output.lower()
        assert "PyPI" in output


class TestTechnicalResearcher:
    """Tests for TechnicalResearcher."""

    def test_researcher_initialization(self):
        """Test researcher initializes."""
        researcher = TechnicalResearcher()

        assert researcher.warden is not None
        assert researcher.research_type == "technical"

    def test_validate_query(self):
        """Test query validation."""
        researcher = TechnicalResearcher()

        valid, result = researcher.validate_query("requests python library")
        assert valid is True

        valid, result = researcher.validate_query("weather forecast")
        assert valid is False


class TestPersonaResearcher:
    """Tests for PersonaResearcher."""

    def test_researcher_initialization(self):
        """Test researcher initializes."""
        researcher = PersonaResearcher()

        assert researcher.research_type == "persona"


class TestSovereignResearcher:
    """Tests for SovereignResearcher orchestrator."""

    def test_researcher_initialization(self, tmp_path):
        """Test sovereign researcher initializes."""
        researcher = SovereignResearcher(project_root=tmp_path)

        assert researcher.technical is not None
        assert researcher.persona is not None
        assert researcher.warden is not None

    def test_verify_report(self, tmp_path):
        """Test report verification flow."""
        researcher = SovereignResearcher(project_root=tmp_path)

        report = FindingsReport(
            query="test",
            findings=[ResearchFinding(query="test", source_url="https://github.com/")]
        )

        researcher.verify_report(report, approved=True)

        assert report.verified is True
        assert len(researcher._verified_findings) == 1

    def test_verify_report_rejected(self, tmp_path):
        """Test rejected report."""
        researcher = SovereignResearcher(project_root=tmp_path)

        report = FindingsReport(query="test", findings=[])
        researcher.verify_report(report, approved=False)

        assert report.verified is True
        assert len(researcher._verified_findings) == 0

    def test_get_stats(self, tmp_path):
        """Test getting researcher statistics."""
        researcher = SovereignResearcher(project_root=tmp_path)

        stats = researcher.get_stats()

        assert "total_reports" in stats
        assert "verified_findings" in stats
        assert "whitelist_size" in stats

    def test_offline_fallback(self, tmp_path):
        """Test offline fallback message."""
        from saga.core.memory import MythosChapter, MythosLibrary

        researcher = SovereignResearcher(project_root=tmp_path)
        mythos = MythosLibrary()
        mythos.add_chapter(MythosChapter(title="Local Knowledge", summary=""))

        message = researcher.get_offline_fallback(mythos)

        assert "Network unavailable" in message
        assert "Local Knowledge" in message
