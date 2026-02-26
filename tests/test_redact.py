"""Tests for deplodock.redact — secret redaction in text and log records."""

import logging

import deplodock.redact as redact_module
from deplodock.redact import SecretRedactingFilter, redact_secrets


def _reset_cache():
    """Reset the module-level pattern cache so env changes take effect."""
    redact_module._patterns = None


# ── redact_secrets ──────────────────────────────────────────────


def test_redact_secrets_replaces_value(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_SuperSecretToken123")
    _reset_cache()

    text = "Downloading with token hf_SuperSecretToken123 from hub"
    assert redact_secrets(text) == "Downloading with token *** from hub"

    _reset_cache()


def test_redact_secrets_short_values_ignored(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "short")
    _reset_cache()

    text = "Token is short and should not be redacted"
    assert redact_secrets(text) == text

    _reset_cache()


def test_redact_secrets_no_env_vars(monkeypatch):
    for var in redact_module._SECRET_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    _reset_cache()

    text = "Nothing secret here"
    assert redact_secrets(text) == text

    _reset_cache()


def test_redact_secrets_multiple_values(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_TokenAAAA")
    monkeypatch.setenv("CLOUDRIFT_API_KEY", "cr_key_BBBB_long_enough")
    _reset_cache()

    text = "HF=hf_TokenAAAA CR=cr_key_BBBB_long_enough done"
    result = redact_secrets(text)
    assert "hf_TokenAAAA" not in result
    assert "cr_key_BBBB_long_enough" not in result
    assert result == "HF=*** CR=*** done"

    _reset_cache()


# ── SecretRedactingFilter ───────────────────────────────────────


def test_secret_redacting_filter(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_FilterTestToken99")
    _reset_cache()

    filt = SecretRedactingFilter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Using token hf_FilterTestToken99",
        args=None,
        exc_info=None,
    )
    filt.filter(record)
    assert record.msg == "Using token ***"

    _reset_cache()


def test_secret_redacting_filter_with_args(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_ArgsTestToken88")
    _reset_cache()

    filt = SecretRedactingFilter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Token: %s",
        args=("hf_ArgsTestToken88",),
        exc_info=None,
    )
    filt.filter(record)
    assert record.args == ("***",)

    _reset_cache()
