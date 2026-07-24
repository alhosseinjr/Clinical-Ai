"""
Shared LLM output parser.

Consolidates the JSON extraction logic that was previously duplicated
across the NLP, Clinical Reasoning, Guideline Verification, and Drug
Safety agents. Every agent now calls parse_llm_json() instead of
re-implementing the same try/regex/fallback pattern.

IMPORTANT: This parser performs lossy recovery operations:
- Smart-quote normalization mutates string contents to recover malformed JSON
- Markdown fence removal may modify legitimate JSON values containing triple backticks
- Trailing comma repair modifies the original text

These operations are intentional design decisions to handle common LLM output failures.
The parser prioritizes successful extraction over preserving exact original content.

Supported JSON structures:
- Nested objects and arrays
- Strings with escaped quotes and backslashes
- Unicode characters (including emoji)
- Escaped Unicode sequences (\\uXXXX)
- Up to 256 levels of nesting (configurable)

Note: Arrays are extracted only when allow_arrays=True. By default, only
JSON objects are accepted. This matches the expected output shape of most
agents in the pipeline.
"""

import json
import logging
import re
from typing import Any

# Type aliases for clearer annotations
JSONDict = dict[str, Any]
JSONArray = list[Any]
ParsedJSON = JSONDict | JSONArray

# Module-level constants
_MAX_LOG_CHARS = 500
_MAX_NESTING = 256

_SMART_QUOTES = {
    "\u201c": '"',  # "
    "\u201d": '"',  # "
    "\u2018": "'",  # '
    "\u2019": "'",  # '
}

_DELIMITER_PAIRS = {
    "}": "{",
    "]": "[",
}

# Safer regex: only strip fences that appear on their own lines
_MARKDOWN_RE = re.compile(
    r"^\s*```[^\n]*\n|\n?```\s*$",
    re.MULTILINE,
)

_TRAILING_COMMA_RE = re.compile(r",\s*([\]}])")

logger = logging.getLogger(__name__)

# Translation table for smart quote normalization (faster than multiple .replace() calls)
_QUOTE_TRANSLATION_TABLE = str.maketrans(_SMART_QUOTES)


def _normalize_smart_quotes(text: str) -> str:
    """
    Normalize smart/curly quotes to straight quotes.
    
    WARNING: This intentionally mutates string contents to recover malformed JSON
    emitted by some LLMs. This is a lossy operation.
    """
    return text.translate(_QUOTE_TRANSLATION_TABLE)


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences that appear on their own lines."""
    return _MARKDOWN_RE.sub("", text).strip()


def _try_parse(text: str, allow_arrays: bool = False) -> ParsedJSON | None:
    """Attempt to parse JSON text directly. Returns None on failure."""
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
        if allow_arrays and isinstance(data, list):
            return data
        return None
    except json.JSONDecodeError:
        return None


def _failure(default: JSONDict, raw_output: str) -> JSONDict:
    """Build failure response with raw output preserved."""
    return {
        **default,
        "raw_output": raw_output,
    }


def _extract_json_block(text: str, allow_arrays: bool = False) -> str | None:
    """
    Extract the first complete JSON object (or array) from text using stack-based
    delimiter tracking. Handles nested braces and brackets correctly by maintaining
    a stack of open delimiters, respecting strings, and validating matching pairs.
    
    Supports:
    - Nested objects and arrays (up to _MAX_NESTING levels)
    - Strings with escaped quotes and backslashes
    - Unicode characters
    - Validates that opening/closing delimiters match correctly
    
    Args:
        text: Input text potentially containing JSON
        allow_arrays: If True, also extract JSON arrays [...]; otherwise only {...}
    
    Returns:
        Extracted JSON string, or None if not found or malformed
    """
    # Find start of object or array
    obj_start = text.find("{")
    arr_start = text.find("[") if allow_arrays else -1
    
    # Determine which comes first
    if obj_start < 0 and arr_start < 0:
        return None
    
    if obj_start < 0:
        start = arr_start
    elif arr_start < 0:
        start = obj_start
    else:
        start = min(obj_start, arr_start)
    
    # Stack-based tracking for nested delimiters
    stack = []
    in_string = False
    escape_next = False
    
    for i in range(start, len(text)):
        char = text[i]
        
        if escape_next:
            escape_next = False
            continue
        
        # Escape handling only matters inside strings
        if in_string and char == "\\":
            escape_next = True
            continue
        
        if char == '"':
            in_string = not in_string
            continue
        
        if in_string:
            continue
        
        if char in "{[":
            stack.append(char)
            # Protect against excessive nesting
            if len(stack) > _MAX_NESTING:
                logger.warning("JSON nesting depth exceeded maximum (%d)", _MAX_NESTING)
                return None
        elif char in "}]":
            if not stack:
                # Unmatched closing delimiter
                return None
            
            opening = stack.pop()
            
            # Validate that delimiters match correctly
            if opening != _DELIMITER_PAIRS[char]:
                return None
            
            # If we just closed the outermost delimiter, we're done
            if not stack:
                return text[start:i+1]
    
    return None


def _try_extract_and_parse(text: str, allow_arrays: bool = False) -> ParsedJSON | None:
    """
    Attempt to extract JSON block and parse it, with trailing comma repair.
    
    This helper consolidates the extraction + parse + repair logic to avoid duplication.
    Returns None if extraction or parsing fails.
    """
    # Attempt extraction
    extracted = _extract_json_block(text, allow_arrays)
    if not extracted:
        return None
    
    # Attempt direct parse
    if data := _try_parse(extracted, allow_arrays):
        logger.debug("Recovered JSON via extraction")
        return data
    
    # Attempt repair of trailing commas
    repaired = _TRAILING_COMMA_RE.sub(r"\1", extracted)
    if data := _try_parse(repaired, allow_arrays):
        logger.debug("Recovered JSON via trailing comma repair")
        return data
    
    return None


def parse_llm_json(
    raw_output: str, 
    default: JSONDict | None = None,
    allow_arrays: bool = False
) -> ParsedJSON:
    """
    Robust JSON parser for small LLM outputs.

    Progressive repair attempts:
    1. Direct parse
    2. Extract JSON block and parse
    3. Normalize smart quotes and strip markdown fences
    4. Extract JSON block again and parse

    IMPORTANT: This parser performs lossy recovery:
    - Smart-quote normalization mutates string contents
    - Markdown fence removal may modify legitimate JSON values
    - Trailing comma repair modifies the original text
    
    Note: Arrays are extracted only when allow_arrays=True.
    
    Args:
        raw_output: The raw string returned by the LLM.
        default: Fallback dict returned when all parsing attempts fail.
                 If None, a generic error dict is returned instead.
        allow_arrays: If True, also accept JSON arrays as valid output.
                      Default is False (only accept objects).

    Returns:
        Parsed dict (or list if allow_arrays=True), or `default` / error dict on failure.
    """
    if default is None:
        default = {"parse_error": "could not parse model output as JSON"}

    if not isinstance(raw_output, str) or not raw_output.strip():
        logger.debug("Empty or non-string input to parse_llm_json")
        return _failure(default, raw_output)

    text = raw_output.strip()
    
    # Strip UTF-8 BOM
    text = text.lstrip("\ufeff")
    
    # Attempt 1: Direct parse
    if data := _try_parse(text, allow_arrays):
        return data
    
    # Attempt 2: Extract JSON block and parse
    if data := _try_extract_and_parse(text, allow_arrays):
        return data
    
    # Attempt 3: Normalize smart quotes
    # This intentionally mutates string content to recover malformed JSON
    text = _normalize_smart_quotes(text)
    if data := _try_parse(text, allow_arrays):
        logger.debug("Recovered JSON after smart quote normalization")
        return data
    
    # Attempt 4: Strip markdown fences
    text = _strip_markdown_fences(text)
    if data := _try_parse(text, allow_arrays):
        logger.debug("Recovered JSON after markdown cleanup")
        return data
    
    # Attempt 5: Extract JSON block again after cleanup
    if data := _try_extract_and_parse(text, allow_arrays):
        return data
    
    # All attempts failed
    logger.warning(
        "Failed parsing JSON (chars=%d arrays=%s): %.500r",
        len(raw_output),
        allow_arrays,
        raw_output,
    )
    return _failure(default, raw_output)