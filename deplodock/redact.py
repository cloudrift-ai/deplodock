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

# Values registered explicitly (e.g. resolved from a CLI flag that may not be in env).
_explicit_secrets: set[str] = set()


def _collect_secret_values() -> set[str]:
    values = set(_explicit_secrets)
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


def register_secret(value: str) -> None:
    """Register an additional value to be redacted from logs.

    Use at every site that resolves a secret from a CLI flag or other source
    that may not be in os.environ when the redactor first scans (e.g. tokens
    passed via `--hf-token`, `--api-key`).
    """
    global _patterns
    if value and len(value) >= _MIN_SECRET_LENGTH and value not in _explicit_secrets:
        _explicit_secrets.add(value)
        _patterns = None  # invalidate cache so next redaction picks it up


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

    Must be attached to *Handlers*, not Loggers. Logger-level filters only run
    on records originating from that logger; records propagating up from a
    child logger bypass them. Handler-level filters run on every record the
    handler processes, so they catch the propagation path too.

    Use ``install_redaction(handler)`` rather than ``handler.addFilter(...)``
    directly to keep wiring uniform across the codebase.
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


def install_redaction(handler: logging.Handler) -> None:
    """Attach a SecretRedactingFilter to the given handler.

    Always prefer this over Logger.addFilter(): only handler-attached filters
    run for records that propagate up from child loggers.
    """
    handler.addFilter(SecretRedactingFilter())
