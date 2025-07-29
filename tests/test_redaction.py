import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from utils.redaction import redact_pii


def test_redact_email():
    assert redact_pii("Contact test@example.com") == "Contact [REDACTED_EMAIL]"


def test_redact_phone():
    assert redact_pii("Call +420 123 456 789") == "Call [REDACTED_PHONE]"


def test_no_pii():
    text = "Hello world"
    assert redact_pii(text) == text
