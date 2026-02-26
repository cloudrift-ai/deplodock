"""Centralized secret redaction for logs and result files."""

import logging
import os
import re

# Env vars whose values should be redacted from all output
_SECRET_ENV_VARS = [
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "CLOUDRIFT_API_KEY",
    "GCP_SERVICE_ACCOUNT",
    "GOOGLE_APPLICATION_CREDENTIALS",
]

_MIN_SECRET_LENGTH = 8  # skip short values to avoid false positives


def _collect_secret_values() -> set[str]:
    values = set()
    for var in _SECRET_ENV_VARS:
        val = os.environ.get(var, "")
        if len(val) >= _MIN_SECRET_LENGTH:
            values.add(val)
    return values


def _build_patterns(values: set[str]) -> list[re.Pattern]:
    # Sort by length descending so longer values match first
    return [re.compile(re.escape(v)) for v in sorted(values, key=len, reverse=True)]


# Lazy-initialized module cache
_patterns: list[re.Pattern] | None = None


def _get_patterns() -> list[re.Pattern]:
    global _patterns
    if _patterns is None:
        _patterns = _build_patterns(_collect_secret_values())
    return _patterns


def redact_secrets(text: str) -> str:
    """Replace known secret env var values with '***'."""
    for pattern in _get_patterns():
        text = pattern.sub("***", text)
    return text


def _apply(text: str, patterns: list[re.Pattern]) -> str:
    for p in patterns:
        text = p.sub("***", text)
    return text


class SecretRedactingFilter(logging.Filter):
    """Logging filter that replaces secret values in log records with '***'.

    Attaches to the root logger so all handlers benefit.
    Handles both f-string messages (msg is pre-formatted) and
    %-style messages (msg + args).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        patterns = _get_patterns()
        if patterns:
            record.msg = _apply(str(record.msg), patterns)
            if record.args:
                if isinstance(record.args, dict):
                    record.args = {k: _apply(str(v), patterns) if isinstance(v, str) else v for k, v in record.args.items()}
                elif isinstance(record.args, tuple):
                    record.args = tuple(_apply(str(a), patterns) if isinstance(a, str) else a for a in record.args)
        return True
