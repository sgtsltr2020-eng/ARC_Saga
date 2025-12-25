"""
Unit Tests for USMA Phase 7: Memory Maintenance
=================================================

Tests for MemoryJanitor, ColdStorage, GlobalWisdomBridge, and EpochManager.
"""


from saga.core.memory import (
    ArchitecturalDebt,
    ColdStorage,
    EpochManager,
    GlobalWisdomBridge,
    MemoryJanitor,
    MythosChapter,
    MythosLibrary,
    ProjectEpoch,
    SolvedPattern,
)


class TestProjectEpoch:
    """Tests for ProjectEpoch dataclass."""

    def test_epoch_creation(self):
        """Test creating a project epoch."""
        epoch = ProjectEpoch(
            name="The Genesis Era",
            description="Project creation phase"
        )

        assert epoch.name == "The Genesis Era"
        assert epoch.is_active() is True

    def test_epoch_serialization(self):
        """Test epoch to_dict and from_dict."""
        epoch = ProjectEpoch(
            name="Test Epoch",
            description="Testing",
            milestone_events=["First commit", "First release"]
        )

        data = epoch.to_dict()
        restored = ProjectEpoch.from_dict(data)

        assert restored.name == epoch.name
        assert len(restored.milestone_events) == 2


class TestEpochManager:
    """Tests for EpochManager."""

    def test_manager_initialization(self, tmp_path):
        """Test manager initializes."""
        manager = EpochManager(storage_path=tmp_path / "epochs.json")
        assert manager.epochs == []

    def test_create_epoch(self, tmp_path):
        """Test creating an epoch."""
        manager = EpochManager(storage_path=tmp_path / "epochs.json")

        epoch = manager.create_epoch(
            name="The Beginning",
            description="Starting the project"
        )

        assert epoch.name == "The Beginning"
        assert manager.get_current_epoch() == epoch

    def test_epoch_succession(self, tmp_path):
        """Test creating new epoch closes previous."""
        manager = EpochManager(storage_path=tmp_path / "epochs.json")

        epoch1 = manager.create_epoch("Era 1")
        epoch2 = manager.create_epoch("Era 2")

        assert epoch1.end_date is not None  # Closed
        assert epoch2.is_active() is True
        assert manager.get_current_epoch() == epoch2

    def test_get_timeline(self, tmp_path):
        """Test getting epoch timeline."""
        manager = EpochManager(storage_path=tmp_path / "epochs.json")

        manager.create_epoch("Phase 1")
        manager.create_epoch("Phase 2")

        timeline = manager.get_timeline()

        assert len(timeline) == 2
        assert timeline[0]["name"] == "Phase 1"

    def test_link_lore_entry(self, tmp_path):
        """Test linking lore entries to epochs."""
        manager = EpochManager(storage_path=tmp_path / "epochs.json")
        epoch = manager.create_epoch("Test Era")

        result = manager.link_lore_entry(epoch.epoch_id, "lore_123")

        assert result is True
        assert "lore_123" in epoch.lore_entry_ids


class TestColdStorage:
    """Tests for ColdStorage SQLite backend."""

    def test_storage_initialization(self, tmp_path):
        """Test storage initializes database."""
        storage = ColdStorage(tmp_path / "cold.db")

        assert (tmp_path / "cold.db").exists()

    def test_archive_pattern(self, tmp_path):
        """Test archiving a pattern."""
        storage = ColdStorage(tmp_path / "cold.db")

        pattern = SolvedPattern(
            name="Singleton Pattern",
            description="Ensure single instance"
        )

        storage.archive_pattern(pattern, original_source_id="orig_123")

        assert storage.get_archived_pattern_count() == 1

    def test_archive_resolved_debt(self, tmp_path):
        """Test archiving resolved debt."""
        storage = ColdStorage(tmp_path / "cold.db")

        debt = ArchitecturalDebt(
            name="Missing Tests",
            description="No unit tests",
            severity="HIGH"
        )

        storage.archive_resolved_debt(debt, resolution_notes="Tests added")
        # Should not raise

    def test_search_archived_patterns(self, tmp_path):
        """Test searching archived patterns."""
        storage = ColdStorage(tmp_path / "cold.db")

        storage.archive_pattern(SolvedPattern(
            name="Retry Logic",
            description="Implements retries with exponential backoff"
        ))
        storage.archive_pattern(SolvedPattern(
            name="Circuit Breaker",
            description="Prevents cascade failures"
        ))

        results = storage.search_archived_patterns("retry")

        assert len(results) == 1
        assert results[0]["name"] == "Retry Logic"


class TestMemoryJanitor:
    """Tests for MemoryJanitor compression."""

    def test_janitor_initialization(self):
        """Test janitor initializes."""
        mythos = MythosLibrary()
        janitor = MemoryJanitor(mythos)

        assert janitor.mythos is mythos

    def test_estimate_token_count(self):
        """Test token estimation."""
        mythos = MythosLibrary()
        mythos.add_chapter(MythosChapter(
            title="Test Chapter",
            summary="A" * 1000  # 1000 chars
        ))

        janitor = MemoryJanitor(mythos)
        tokens = janitor.estimate_token_count()

        assert tokens > 0

    def test_needs_compression(self):
        """Test compression threshold detection."""
        mythos = MythosLibrary()
        janitor = MemoryJanitor(mythos, token_threshold=100)

        # Small mythos
        assert janitor.needs_compression() is False

        # Add large content
        mythos.add_chapter(MythosChapter(
            title="Big Chapter",
            summary="A" * 5000
        ))

        assert janitor.needs_compression() is True

    def test_find_duplicate_patterns(self):
        """Test finding duplicate patterns."""
        mythos = MythosLibrary()
        mythos.add_chapter(MythosChapter(
            title="Chapter 1",
            summary="",
            solved_patterns=[
                SolvedPattern(name="Retry Logic", description="Retry with backoff"),
                SolvedPattern(name="Retry Pattern", description="Retry with backoff exponential")
            ]
        ))

        janitor = MemoryJanitor(mythos, similarity_threshold=0.5)
        duplicates = janitor.find_duplicate_patterns()

        assert len(duplicates) >= 1

    def test_compress_mythos(self, tmp_path):
        """Test mythos compression."""
        mythos = MythosLibrary()
        mythos.add_chapter(MythosChapter(
            title="Test",
            summary="",
            solved_patterns=[
                SolvedPattern(name="Pattern A", description="Does thing A"),
                SolvedPattern(name="Pattern A Same", description="Does thing A exactly")
            ]
        ))

        storage = ColdStorage(tmp_path / "cold.db")
        janitor = MemoryJanitor(mythos, cold_storage=storage, similarity_threshold=0.5)

        result = janitor.compress_mythos()

        assert result["patterns_before"] == 2
        assert result["duplicates_removed"] >= 0


class TestGlobalWisdomBridge:
    """Tests for GlobalWisdomBridge multi-project sync."""

    def test_bridge_initialization(self, tmp_path):
        """Test bridge initializes."""
        bridge = GlobalWisdomBridge(global_path=tmp_path / "global_wisdom")

        assert bridge.global_path.exists()

    def test_export_pattern(self, tmp_path):
        """Test exporting pattern to global wisdom."""
        bridge = GlobalWisdomBridge(global_path=tmp_path / "global_wisdom")

        pattern = SolvedPattern(
            name="Universal Retry",
            description="Works everywhere"
        )

        result = bridge.export_pattern(
            pattern=pattern,
            source_project="ProjectA",
            utility_score=0.9
        )

        assert result is True
        assert bridge.get_pattern_count() == 1

    def test_import_patterns(self, tmp_path):
        """Test importing patterns from global wisdom."""
        bridge = GlobalWisdomBridge(global_path=tmp_path / "global_wisdom")

        # Export some patterns
        bridge.export_pattern(
            SolvedPattern(name="High Utility", description="Very useful"),
            source_project="ProjectA",
            utility_score=0.9
        )
        bridge.export_pattern(
            SolvedPattern(name="Low Utility", description="Not as useful"),
            source_project="ProjectB",
            utility_score=0.5
        )

        # Import high-utility only
        imported = bridge.import_patterns(min_utility=0.7)

        assert len(imported) == 1
        assert "[Global]" in imported[0].name

    def test_pattern_sanitization(self, tmp_path):
        """Test patterns are sanitized on export."""
        bridge = GlobalWisdomBridge(global_path=tmp_path / "global_wisdom")

        pattern = SolvedPattern(
            name="API Pattern",
            description="Uses api_key=sk_live_secret123 for auth"
        )

        bridge.export_pattern(pattern, "TestProject", 0.8)

        # Check that sensitive data was sanitized
        imported = bridge.import_patterns(min_utility=0.5)
        assert "sk_live" not in imported[0].description
        assert "[REDACTED]" in imported[0].description

    def test_duplicate_handling(self, tmp_path):
        """Test duplicate patterns are updated, not duplicated."""
        bridge = GlobalWisdomBridge(global_path=tmp_path / "global_wisdom")

        pattern = SolvedPattern(name="Same Pattern", description="Version 1")
        bridge.export_pattern(pattern, "ProjectA", utility_score=0.7)

        pattern2 = SolvedPattern(name="Same Pattern", description="Version 2 better")
        bridge.export_pattern(pattern2, "ProjectB", utility_score=0.9)

        assert bridge.get_pattern_count() == 1  # Not duplicated

    def test_clear_global_wisdom(self, tmp_path):
        """Test clearing global wisdom."""
        bridge = GlobalWisdomBridge(global_path=tmp_path / "global_wisdom")

        bridge.export_pattern(
            SolvedPattern(name="Test", description=""),
            "Project",
            0.8
        )

        bridge.clear_global_wisdom()

        assert bridge.get_pattern_count() == 0
