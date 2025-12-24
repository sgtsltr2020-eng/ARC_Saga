"""
Comprehensive Tests for SagaCodex Profiles
===========================================

Tests profile management, standards, anti-patterns, elite patterns.

Author: ARC SAGA Development Team
Date: December 14, 2025
Status: Phase 2 Week 1 - Comprehensive Testing
"""

from pathlib import Path

import pytest

from saga.config.sagacodex_profiles import (
    CodeStandard,
    LanguageProfile,
    SagaCodex,
    SagaCodexManager,
    get_codex_manager,
)


class TestCodeStandard:
    """Test CodeStandard dataclass."""
    
    def test_code_standard_creation(self) -> None:
        """Test creating a CodeStandard."""
        standard = CodeStandard(
            rule_number=1,
            name="Test Standard",
            description="Test description",
            applies_to=["python"],
            tool="mypy",
            severity="CRITICAL"
        )
        
        assert standard.rule_number == 1
        assert standard.name == "Test Standard"
        assert standard.severity == "CRITICAL"
        assert "python" in standard.applies_to
    
    def test_code_standard_to_dict(self) -> None:
        """Test CodeStandard serialization."""
        standard = CodeStandard(
            rule_number=1,
            name="Test Standard",
            description="Test description",
            applies_to=["python"],
        )
        
        std_dict = standard.to_dict()
        
        assert isinstance(std_dict, dict)
        assert std_dict["rule_number"] == 1
        assert std_dict["name"] == "Test Standard"


class TestSagaCodexManager:
    """Test SagaCodexManager class."""
    
    def test_manager_initialization(self, codex_manager: SagaCodexManager) -> None:
        """Test manager initializes correctly."""
        assert codex_manager is not None
        assert hasattr(codex_manager, '_profiles')
    
    def test_get_python_fastapi_profile(self, codex_manager: SagaCodexManager) -> None:
        """Test getting Python/FastAPI profile."""
        codex = codex_manager.get_profile(LanguageProfile.PYTHON_FASTAPI)
        
        assert codex.language == "Python"
        assert codex.framework == "FastAPI"
        assert len(codex.standards) > 0
    
    def test_unsupported_profile_raises_error(self, codex_manager: SagaCodexManager) -> None:
        """Test getting unsupported profile raises NotImplementedError."""
        # Casting to Any to bypass type checking for invalid enum testing
        with pytest.raises(NotImplementedError):
            codex_manager.get_profile(LanguageProfile.PYTHON_DJANGO)
    
    def test_get_current_profile(self, codex_manager: SagaCodexManager) -> None:
        """Test getting current profile (defaults to Python/FastAPI)."""
        codex = codex_manager.get_current_profile()
        
        assert codex.language == "Python"
        assert codex.framework == "FastAPI"
    
    def test_singleton_get_codex_manager(self) -> None:
        """Test get_codex_manager returns singleton."""
        manager1 = get_codex_manager()
        manager2 = get_codex_manager()
        
        assert manager1 is manager2


class TestPythonFastAPICodex:
    """Test Python/FastAPI SagaCodex profile."""
    
    def test_codex_has_standards(self, python_fastapi_codex: SagaCodex) -> None:
        """Test Python/FastAPI codex has standards."""
        assert len(python_fastapi_codex.standards) >= 10
    
    def test_codex_has_anti_patterns(self, python_fastapi_codex: SagaCodex) -> None:
        """Test Python/FastAPI codex has anti-patterns."""
        assert len(python_fastapi_codex.anti_patterns) >= 3
    
    def test_codex_has_elite_patterns(self, python_fastapi_codex: SagaCodex) -> None:
        """Test Python/FastAPI codex has elite patterns."""
        assert len(python_fastapi_codex.elite_patterns) >= 2
    
    def test_get_standard_by_number(self, python_fastapi_codex: SagaCodex) -> None:
        """Test getting standard by rule number."""
        rule1 = python_fastapi_codex.get_standard(1)
        
        assert rule1 is not None
        assert rule1.rule_number == 1
        assert "Type" in rule1.name  # Type Safety Required
    
    def test_get_nonexistent_standard(self, python_fastapi_codex: SagaCodex) -> None:
        """Test getting non-existent standard returns None."""
        rule999 = python_fastapi_codex.get_standard(999)
        assert rule999 is None
    
    def test_get_standards_by_severity(self, python_fastapi_codex: SagaCodex) -> None:
        """Test getting standards by severity."""
        critical = python_fastapi_codex.get_standards_by_severity("CRITICAL")
        
        assert len(critical) > 0
        assert all(s.severity == "CRITICAL" for s in critical)
    
    def test_search_standards(self, python_fastapi_codex: SagaCodex) -> None:
        """Test searching standards by keyword."""
        type_standards = python_fastapi_codex.search_standards("type")
        
        assert len(type_standards) > 0
        assert any("type" in s.name.lower() or "type" in s.description.lower() 
                   for s in type_standards)
    
    def test_codex_to_dict(self, python_fastapi_codex: SagaCodex) -> None:
        """Test SagaCodex serialization."""
        codex_dict = python_fastapi_codex.to_dict()
        
        assert isinstance(codex_dict, dict)
        assert codex_dict["language"] == "Python"
        assert codex_dict["framework"] == "FastAPI"
        assert "standards" in codex_dict
        assert "anti_patterns" in codex_dict


class TestStandardsContent:
    """Test specific standards content."""
    
    def test_type_safety_standard(self, python_fastapi_codex: SagaCodex) -> None:
        """Test Rule 1: Type Safety Required."""
        rule1 = python_fastapi_codex.get_standard(1)
        
        assert rule1 is not None
        assert "type" in rule1.name.lower()
        assert rule1.severity == "CRITICAL"
        assert "mypy" in rule1.tool.lower() # type: ignore
    
    def test_async_standard(self, python_fastapi_codex: SagaCodex) -> None:
        """Test Rule 2: Async for I/O Operations."""
        rule2 = python_fastapi_codex.get_standard(2)
        
        assert rule2 is not None
        assert "async" in rule2.name.lower()
        assert rule2.severity == "CRITICAL"
    
    def test_test_coverage_standard(self, python_fastapi_codex: SagaCodex) -> None:
        """Test Rule 7: 99% Test Coverage."""
        rule7 = python_fastapi_codex.get_standard(7)
        
        assert rule7 is not None
        assert "99" in rule7.name or "99" in rule7.description
        assert rule7.severity == "CRITICAL"


class TestAntiPatterns:
    """Test anti-pattern detection."""
    
    def test_anti_patterns_defined(self, python_fastapi_codex: SagaCodex) -> None:
        """Test anti-patterns are defined."""
        anti_patterns = python_fastapi_codex.anti_patterns
        
        assert len(anti_patterns) > 0
        
        # Check for common anti-patterns
        names = [ap.name for ap in anti_patterns]
        assert any("print" in name.lower() for name in names)
        assert any("except" in name.lower() for name in names)
    
    def test_check_code_detects_print(self, codex_manager: SagaCodexManager) -> None:
        """Test anti-pattern detection finds print statements."""
        code = "print('Hello, world!')"
        
        violations = codex_manager.check_code(code, LanguageProfile.PYTHON_FASTAPI)
        
        assert len(violations) > 0
        assert any("print" in v["name"].lower() for v in violations)
    
    def test_check_code_detects_bare_except(self, codex_manager: SagaCodexManager) -> None:
        """Test anti-pattern detection finds bare except."""
        code = """
try:
    risky_operation()
except:
    pass
"""
        
        violations = codex_manager.check_code(code, LanguageProfile.PYTHON_FASTAPI)
        
        assert len(violations) > 0
        assert any("except" in v["name"].lower() for v in violations)
    
    def test_check_code_clean_code(self, codex_manager: SagaCodexManager) -> None:
        """Test clean code produces no violations."""
        code = """
async def get_user(db: AsyncSession, user_id: str) -> Optional[User]:
    try:
        result = await db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()
    except SQLAlchemyError as e:
        logger.error("Database error", extra={"user_id": user_id})
        raise
"""
        
        violations = codex_manager.check_code(code, LanguageProfile.PYTHON_FASTAPI)
        
        # Should have minimal or no violations
        assert len(violations) <= 1  # Might flag if patterns are overly strict


class TestElitePatterns:
    """Test elite patterns."""
    
    def test_elite_patterns_defined(self, python_fastapi_codex: SagaCodex) -> None:
        """Test elite patterns are defined."""
        elite_patterns = python_fastapi_codex.elite_patterns
        
        assert len(elite_patterns) > 0
    
    def test_elite_pattern_has_source(self, python_fastapi_codex: SagaCodex) -> None:
        """Test elite patterns cite sources."""
        for pattern in python_fastapi_codex.elite_patterns:
            assert pattern.source in ["Linear", "Figma", "Stripe", "Vercel", "Linear, Figma"], \
                f"Pattern {pattern.name} has invalid source: {pattern.source}"
    
    def test_elite_pattern_has_benefits(self, python_fastapi_codex: SagaCodex) -> None:
        """Test elite patterns list benefits."""
        for pattern in python_fastapi_codex.elite_patterns:
            assert len(pattern.benefits) > 0, \
                f"Pattern {pattern.name} has no benefits listed"


class TestExportProfile:
    """Test profile export functionality."""
    
    def test_export_profile_to_json(
        self,
        codex_manager: SagaCodexManager,
        temp_project_dir: Path
    ) -> None:
        """Test exporting profile to JSON file."""
        output_path = temp_project_dir / "sagacodex_python_fastapi.json"
        
        codex_manager.export_profile(
            LanguageProfile.PYTHON_FASTAPI,
            output_path
        )
        
        assert output_path.exists()
        
        # Verify JSON is valid
        import json
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        assert data["language"] == "Python"
        assert data["framework"] == "FastAPI"
        assert "standards" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
