"""Tests for emmy.redact — secret redaction in text and log records."""

import logging

import emmy.redact as redact_module
from emmy.redact import SecretRedactingFilter, install_redaction, redact_secrets, register_secret


def _reset_cache():
    """Reset the module-level pattern cache and explicit-secrets set."""
    redact_module._patterns = None
    redact_module._explicit_secrets.clear()


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


# ── install_redaction: end-to-end through a real handler ─────────


def test_install_redaction_through_file_handler(monkeypatch, tmp_path):
    """Regression: child-logger records propagating to a file handler must
    still be redacted. Logger-level filters miss this path; handler-level
    filters (what install_redaction installs) catch it.
    """
    monkeypatch.setenv("HF_TOKEN", "hf_PropagationTestToken123")
    _reset_cache()

    log_file = tmp_path / "out.log"
    handler = logging.FileHandler(log_file, encoding="utf-8")
    install_redaction(handler)

    root = logging.getLogger()
    prev_level = root.level
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    try:
        # Emit from a *child* logger to exercise propagation through callHandlers.
        child = logging.getLogger("emmy.provisioning.ssh_transport")
        child.info("[dry-run] ssh host: docker run -e HUGGING_FACE_HUB_TOKEN=hf_PropagationTestToken123 ...")
        handler.flush()
    finally:
        root.removeHandler(handler)
        handler.close()
        root.setLevel(prev_level)

    contents = log_file.read_text()
    assert "hf_PropagationTestToken123" not in contents
    assert "***" in contents

    _reset_cache()


# ── register_secret: values not in env ──────────────────────────


def test_register_secret_redacts_value_not_in_env(monkeypatch):
    for var in redact_module._SECRET_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    _reset_cache()

    register_secret("cli_only_secret_AAAAAA")
    assert redact_secrets("token=cli_only_secret_AAAAAA done") == "token=*** done"

    _reset_cache()


def test_register_secret_short_values_ignored(monkeypatch):
    for var in redact_module._SECRET_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    _reset_cache()

    register_secret("short")
    assert redact_secrets("value=short here") == "value=short here"

    _reset_cache()


def test_register_secret_invalidates_cache(monkeypatch):
    for var in redact_module._SECRET_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    _reset_cache()

    # Prime the cache while no secrets are known.
    assert redact_secrets("placeholder") == "placeholder"

    register_secret("late_registered_value_BBBB")
    # Cache should have been invalidated; the new value is now redacted.
    assert redact_secrets("v=late_registered_value_BBBB") == "v=***"

    _reset_cache()


def test_register_secret_propagates_through_file_handler(monkeypatch, tmp_path):
    """A value registered via register_secret() (e.g. from --hf-token / --api-key
    that wasn't in os.environ) is still redacted from file-handler output.
    """
    for var in redact_module._SECRET_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    _reset_cache()

    register_secret("cr_cli_flag_secret_CCCCCC")

    log_file = tmp_path / "out.log"
    handler = logging.FileHandler(log_file, encoding="utf-8")
    install_redaction(handler)

    root = logging.getLogger()
    prev_level = root.level
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    try:
        child = logging.getLogger("emmy.provisioning.cloudrift")
        child.info("X-API-Key: cr_cli_flag_secret_CCCCCC")
        handler.flush()
    finally:
        root.removeHandler(handler)
        handler.close()
        root.setLevel(prev_level)

    contents = log_file.read_text()
    assert "cr_cli_flag_secret_CCCCCC" not in contents
    assert "***" in contents

    _reset_cache()
