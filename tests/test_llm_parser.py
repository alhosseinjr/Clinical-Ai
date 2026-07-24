"""
tests/test_llm_parser.py

Unit tests for the shared LLM JSON parser.
"""

import pytest
from src.utils.llm_parser import parse_llm_json


class TestParseLLMJson:
    """Test suite for parse_llm_json function."""

    def test_valid_json(self):
        """Pure JSON should parse directly."""
        result = parse_llm_json('{"a": 1, "b": "test"}')
        assert result == {"a": 1, "b": "test"}

    def test_json_in_markdown_fences(self):
        """JSON wrapped in ```json fences should be extracted."""
        result = parse_llm_json('```json\n{"a": 1}\n```')
        assert result == {"a": 1}

    def test_json_in_markdown_fences_with_language(self):
        """JSON wrapped in ```javascript or other language tags."""
        result = parse_llm_json('```javascript\n{"a": 1}\n```')
        assert result == {"a": 1}

    def test_json_in_markdown_fences_with_space(self):
        """JSON wrapped in ``` json (with space) should be extracted."""
        result = parse_llm_json('``` json\n{"a": 1}\n```')
        assert result == {"a": 1}

    def test_json_in_markdown_fences_uppercase(self):
        """JSON wrapped in ```JSON (uppercase) should be extracted."""
        result = parse_llm_json('```JSON\n{"a": 1}\n```')
        assert result == {"a": 1}

    def test_nested_objects(self):
        """Nested JSON objects should parse correctly."""
        result = parse_llm_json('{"outer": {"inner": {"value": 42}}}')
        assert result == {"outer": {"inner": {"value": 42}}}

    def test_nested_arrays(self):
        """Nested JSON arrays should parse correctly."""
        result = parse_llm_json('{"a": [{"x": 1}, {"y": 2}]}')
        assert result == {"a": [{"x": 1}, {"y": 2}]}

    def test_empty_object(self):
        """Empty JSON object should parse."""
        result = parse_llm_json('{}')
        assert result == {}

    def test_empty_array(self):
        """Empty JSON array should parse when allowed."""
        result = parse_llm_json('[]', allow_arrays=True)
        assert result == []

    def test_utf8_bom(self):
        """UTF-8 BOM should be stripped."""
        result = parse_llm_json('\ufeff{"a": 1}')
        assert result == {"a": 1}

    def test_smart_quotes_in_keys(self):
        """Smart/curly quotes in keys should be normalized."""
        result = parse_llm_json('{\u201ctext\u201d: \u201chello\u201d}')
        assert result == {"text": "hello"}

    def test_smart_quotes_in_values(self):
        """Smart/curly quotes in values should be normalized."""
        result = parse_llm_json('{"text": "He said \u201cHello\u201d"}')
        assert result == {"text": 'He said "Hello"'}

    def test_trailing_commas(self):
        """Trailing commas should be repaired."""
        result = parse_llm_json('{"a": 1, "b": 2,}')
        assert result == {"a": 1, "b": 2}

    def test_trailing_commas_in_nested(self):
        """Trailing commas in nested objects should be repaired."""
        result = parse_llm_json('{"a": {"b": 1,},}')
        assert result == {"a": {"b": 1}}

    def test_extra_text_before_json(self):
        """Extra prose before JSON should be ignored."""
        result = parse_llm_json('Here is the result:\n{"a": 1}')
        assert result == {"a": 1}

    def test_extra_text_after_json(self):
        """Extra prose after JSON should be ignored."""
        result = parse_llm_json('{"a": 1}\nDone.')
        assert result == {"a": 1}

    def test_extra_text_before_and_after(self):
        """Extra prose before and after JSON should be ignored."""
        result = parse_llm_json('Sure!\n{"a": 1}\nHope this helps!')
        assert result == {"a": 1}

    def test_braces_inside_string_values(self):
        """Braces inside string values should not break extraction."""
        result = parse_llm_json('{"text": "Patient said {hello}", "risk": {"score": 0.82}}')
        assert result == {"text": "Patient said {hello}", "risk": {"score": 0.82}}

    def test_empty_string(self):
        """Empty string should return default."""
        result = parse_llm_json('')
        assert "parse_error" in result
        assert result["raw_output"] == ""

    def test_none_input(self):
        """None input should return default."""
        result = parse_llm_json(None)
        assert "parse_error" in result

    def test_non_string_input(self):
        """Non-string input should return default."""
        result = parse_llm_json(123)
        assert "parse_error" in result

    def test_completely_invalid_output(self):
        """Completely invalid output should return default."""
        result = parse_llm_json('This is not JSON at all')
        assert "parse_error" in result
        assert result["raw_output"] == 'This is not JSON at all'

    def test_custom_default(self):
        """Custom default should be returned on failure."""
        custom_default = {"status": "failed", "data": None}
        result = parse_llm_json('invalid', default=custom_default)
        assert result["status"] == "failed"
        assert result["data"] is None
        assert result["raw_output"] == 'invalid'

    def test_list_instead_of_dict(self):
        """JSON array should return default when allow_arrays=False."""
        result = parse_llm_json('[1, 2, 3]')
        assert "parse_error" in result

    def test_list_when_allowed(self):
        """JSON array should parse when allow_arrays=True."""
        result = parse_llm_json('[1, 2, 3]', allow_arrays=True)
        assert result == [1, 2, 3]

    def test_multiple_json_objects(self):
        """Multiple JSON objects should extract the first complete one."""
        result = parse_llm_json('{"a": 1}\n\n{"b": 2}')
        assert result == {"a": 1}

    def test_valid_json_followed_by_garbage(self):
        """Valid JSON followed by incomplete JSON should extract the first one."""
        result = parse_llm_json('{"a":1}\ngarbage\n{')
        assert result == {"a": 1}

    def test_multiple_fenced_blocks(self):
        """Multiple fenced blocks should extract JSON from the first valid one."""
        result = parse_llm_json('```\ntext\n```\n{"a": 1}\n```json\n{"b": 2}\n```')
        assert result == {"a": 1}

    def test_complex_nested_structure(self):
        """Complex nested structure with arrays and objects."""
        json_str = '''
        {
            "patient": {
                "age": 65,
                "conditions": ["hypertension", "diabetes"]
            },
            "risk": {
                "score": 0.82,
                "category": "high"
            }
        }
        '''
        result = parse_llm_json(json_str)
        assert result["patient"]["age"] == 65
        assert "hypertension" in result["patient"]["conditions"]
        assert result["risk"]["score"] == 0.82

    def test_escaped_quotes_in_strings(self):
        """Escaped quotes inside strings should be handled correctly."""
        result = parse_llm_json('{"text": "He said \\"hello\\""}')
        assert result == {"text": 'He said "hello"'}

    def test_whitespace_only(self):
        """Whitespace-only input should return default."""
        result = parse_llm_json('   \n\t  ')
        assert "parse_error" in result

    def test_json_with_prose_and_fences(self):
        """JSON with prose and markdown fences should parse."""
        result = parse_llm_json('```\nblah blah\n\n{"a": 1}\n```')
        assert result == {"a": 1}

    def test_deeply_nested_with_trailing_commas(self):
        """Deeply nested JSON with trailing commas."""
        result = parse_llm_json('{"a": {"b": {"c": 1,},},}')
        assert result == {"a": {"b": {"c": 1}}}

    def test_array_with_trailing_comma(self):
        """Array with trailing comma when allowed."""
        result = parse_llm_json('[1, 2, 3,]', allow_arrays=True)
        assert result == [1, 2, 3]

    def test_array_containing_objects(self):
        """Array containing objects should parse when allowed."""
        result = parse_llm_json('[{"a": 1}, {"b": 2}]', allow_arrays=True)
        assert result == [{"a": 1}, {"b": 2}]

    def test_object_containing_arrays(self):
        """Object containing arrays should parse correctly."""
        result = parse_llm_json('{"items": [1, 2, 3], "nested": [{"x": 1}]}')
        assert result == {"items": [1, 2, 3], "nested": [{"x": 1}]}