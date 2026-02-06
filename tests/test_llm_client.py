from __future__ import annotations
import pytest
from memalign_mcp.llm_client import parse_json_response


class TestParseJsonResponse:
    def test_raw_json(self):
        result = parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_with_code_fence(self):
        text = '```json\n{"key": "value"}\n```'
        result = parse_json_response(text)
        assert result == {"key": "value"}

    def test_json_with_plain_fence(self):
        text = '```\n{"key": "value"}\n```'
        result = parse_json_response(text)
        assert result == {"key": "value"}

    def test_json_with_whitespace(self):
        result = parse_json_response('  \n  {"key": "value"}  \n  ')
        assert result == {"key": "value"}

    def test_json_embedded_in_text(self):
        text = 'Here is the result: {"key": "value"} end'
        result = parse_json_response(text)
        assert result == {"key": "value"}

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Could not parse"):
            parse_json_response("not json at all")

    def test_non_dict_json_raises(self):
        with pytest.raises(ValueError, match="Expected JSON object"):
            parse_json_response("[1, 2, 3]")

    def test_nested_json(self):
        text = '{"principles": [{"text": "be safe"}], "reasoning": "because"}'
        result = parse_json_response(text)
        assert len(result["principles"]) == 1
        assert result["principles"][0]["text"] == "be safe"

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            parse_json_response("")
