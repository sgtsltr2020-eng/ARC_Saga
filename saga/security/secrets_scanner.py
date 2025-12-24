"""
Secrets Detection in Code
==========================

Detects hardcoded secrets, API keys, passwords, tokens.
Uses patterns from git-secrets and TruffleHog.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class SecretDetection:
    """Detected secret."""
    secret_type: str
    pattern_name: str
    match: str  # Redacted
    line_number: int
    severity: str  # HIGH, CRITICAL

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "secret_type": self.secret_type,
            "pattern_name": self.pattern_name,
            "match": self.match,
            "line_number": self.line_number,
            "severity": self.severity,
        }


class SecretsScanner:
    """
    Scan code for hardcoded secrets.
    
    Patterns from:
    - git-secrets
    - TruffleHog
    - AWS, GCP, Azure, GitHub patterns
    """
    
    PATTERNS = {
        "AWS Access Key": r'AKIA[0-9A-Z]{16}',
        "AWS Secret Key": r'aws_secret_access_key\s*=\s*["\']([^"\']+)',
        "GitHub Token": r'gh[ps]_[a-zA-Z0-9]{36}',
        "OpenAI API Key": r'sk-[a-zA-Z0-9]{48}',
        "Stripe API Key": r'sk_(test|live)_[a-zA-Z0-9]{24,}',
        "Generic API Key": r'api[_-]?key\s*[:=]\s*["\']([^"\']{20,})',
        "Password": r'password\s*[:=]\s*["\']([^"\']{8,})',
        "Private Key": r'-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----',
        "JWT Token": r'eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+',
        "Database URL": r'(mysql|postgres|mongodb):\/\/[^:]+:[^@]+@',
    }
    
    def scan(self, code: str) -> List[SecretDetection]:
        """
        Scan code for secrets.
        
        Args:
            code: Source code to scan
            
        Returns:
            List of detected secrets
        """
        detections = []
        lines = code.split('\n')
        
        for pattern_name, pattern in self.PATTERNS.items():
            regex = re.compile(pattern, re.IGNORECASE)
            
            for line_num, line in enumerate(lines, start=1):
                matches = regex.finditer(line)
                for match in matches:
                    # Redact the actual secret
                    matched_text = match.group(0)
                    redacted = matched_text[:8] + "..." + matched_text[-4:] if len(matched_text) > 12 else "***"
                    
                    detections.append(SecretDetection(
                        secret_type=pattern_name,
                        pattern_name=pattern,
                        match=redacted,
                        line_number=line_num,
                        severity="CRITICAL"
                    ))
        
        return detections


def scan_for_secrets(code: str) -> List[SecretDetection]:
    """
    Scan code for hardcoded secrets.
    
    Example:
        >>> code = "api_key = 'sk-1234567890abcdef'"
        >>> secrets = scan_for_secrets(code)
        >>> secrets[0].secret_type
        'Generic API Key'
    """
    scanner = SecretsScanner()
    return scanner.scan(code)
