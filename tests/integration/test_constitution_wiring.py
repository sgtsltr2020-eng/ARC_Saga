"""
Integration Test for SagaConstitution Wiring
============================================

Verifies that SagaConstitution correctly triggers the Debate Protocol.
"""

import pytest

from saga.config.sagarules_embedded import EscalationContext, SagaConstitution
from saga.core.debate_manager import TriggerSeverity, TriggerType, get_debate_manager
from saga.core.exceptions import SagaEscalationException


def test_constitution_wiring_secrets():
    """Test that secrets detection triggers critical debate."""
    manager = get_debate_manager()
    manager.clear_pending()

    # Create triggering context
    context = EscalationContext(
        secrets_detected=True,
    )

    # Verify exception is raised
    with pytest.raises(SagaEscalationException) as excinfo:
        SagaConstitution.check_and_escalate(context, "Show me keys")

    # Verify exception contains request_id
    request_id = excinfo.value.request_id
    assert request_id is not None

    # Verify request was created in manager
    assert manager.has_pending_requests()
    request = manager.get_request(request_id)
    assert request is not None
    assert request.trigger_type == TriggerType.CONSTITUTION
    assert request.trigger_severity == TriggerSeverity.CRITICAL
    assert "Secrets detected" in request.trigger_description
    assert "Rule 12" in request.violated_rules[0]

def test_constitution_wiring_low_confidence():
    """Test that low confidence triggers warning debate."""
    manager = get_debate_manager()
    manager.clear_pending()

    context = EscalationContext(
        saga_confidence=50.0,
    )

    with pytest.raises(SagaEscalationException) as excinfo:
        SagaConstitution.check_and_escalate(context, "Do complex thing")

    request_id = excinfo.value.request_id
    request = manager.get_request(request_id)
    assert request.trigger_type == TriggerType.CONFIDENCE
    assert request.trigger_severity == TriggerSeverity.WARNING
    assert "Confidence" in request.trigger_description

def test_constitution_no_trigger():
    """Test that safe context does not trigger debate."""
    manager = get_debate_manager()
    manager.clear_pending()

    context = EscalationContext(
        saga_confidence=99.0,
    )

    # Should not raise exception
    SagaConstitution.check_and_escalate(context, "Safe task")

    assert not manager.has_pending_requests()
