import json
import logging

from evidenceflow.core.logging import JsonFormatter


def test_json_formatter_includes_custom_fields() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "hello", (), None)
    record.request_id = "req-1"
    payload = json.loads(formatter.format(record))
    assert payload["level"] == "INFO"
    assert payload["message"] == "hello"
    assert payload["request_id"] == "req-1"
